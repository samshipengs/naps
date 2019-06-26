import os
import re
import time
import pprint
import pandas as pd
import numpy as np
from datetime import datetime as dt
from functools import partial

from ast import literal_eval
from tqdm import tqdm
from sklearn.model_selection import KFold, ShuffleSplit
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import optimizers
from keras.utils import plot_model
from keras.models import load_model
from keras.models import Model
import tensorflow as tf
import lightgbm as lgb
import xgboost as xgb

from model import build_model
from clr_callback import CyclicLR
from create_model_inputs import create_model_inputs, expand
from utils import get_logger, get_data_path
from plots import plot_hist, confusion_matrix


logger = get_logger('train_nn')
Filepath = get_data_path()
RS = 42


# def custom_objective(y_true, y_pred):
#     y_pred_value = K.sum(y_true * y_pred, axis=-1)
#     delta = K.maximum(0., y_pred - y_pred_value[:, None])
#     twos = 2**delta
#     mask = K.cast(K.greater(twos-1, 0), 'float32')
#     sums = K.sum(twos*mask, axis=-1)
#     return K.mean(sums)
def evalmetric(preds, dtrain):
    labels = dtrain.get_label()
    pred_label = np.where(np.argsort(preds)[:, ::-1] == labels.reshape(-1, 1))[1]
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'mrr', 1 - np.mean(1 / (pred_label + 1))


def lgb_mrr(y_preds, train_data):
    y_true = train_data.get_label()
    pred_label = np.where((np.argsort(y_preds.reshape(25, -1), axis=0)[::-1, :] == y_true).T)[1]
    return 'mrr', np.mean(1 / (pred_label + 1)), True


def bpr_max_loss(l2=True, weight=1.):
    def bpr_max(y_true, y_pred):
        # get the predicted value for target
        y_pred_value = K.sum(y_true * y_pred, axis=-1)
        # compute softmax
        softmax = K.softmax(y_pred, axis=-1)
        # compute the margin between predicted score and the rest
        delta = y_pred_value[:, None] - y_pred
        # affply sigmoid over the difference
        sig_delta = K.sigmoid(delta)
        # get the mask of non-target (or negative targets)
        negative_mask = K.cast(K.equal(y_true, 0), 'float32')
        # positive_mask = K.cast(K.equal(y_true, 1), 'float32')

        # get the sum of product over negative targets (and then sum over batch)
        sum_product = softmax * sig_delta * negative_mask
        # apply negative log
        neg_logs = -K.mean(K.log(K.sum(sum_product, axis=-1)))
        # add regularizer which also pushes the negative scores down
        if l2:
            reg = K.mean(K.sum(softmax * y_pred**2 * negative_mask, axis=-1))
        else:
            reg = K.mean(K.sum(softmax * K.abs(y_pred) * negative_mask, axis=-1))
        return neg_logs + weight * reg
    return bpr_max


custom_objective = bpr_max_loss(l2=True, weight=1.)


def iterate_minibatches(input_x, targets, batch_size, shuffle=True):
    indices = np.arange(len(targets))
    while True:
        if shuffle:
            np.random.shuffle(indices)

        remainder = len(targets) % batch_size
        for start_idx in range(0, len(targets), batch_size):
            if remainder != 0 and start_idx + batch_size >= len(targets):
                excerpt = indices[len(targets) - batch_size:len(targets)]
            else:
                excerpt = indices[start_idx:start_idx + batch_size]

            input_batch = input_x[excerpt]
            targets_batch = targets[excerpt]
            yield (input_batch, targets_batch)


def log_median(df, col):
    df[col] = np.log((1+df[col])/(1+np.median(df[col])))


def nn_prep(df):
    # specify some columns that we do not want in training
    cf_cols = [i for i in df.columns if 'current_filters' in i]
    # price_cols = [i for i in df.columns if re.match(r'prices_\d', i)]
    bin_cols = [i for i in df.columns if 'bin' in i]

    drop_cols = cf_cols + bin_cols + ['country', 'platform', 'impressions_str', 'fs', 'sort_order']
    drop_cols = [col for col in df.columns if col in drop_cols]
    # drop col
    logger.info(f'Preliminary Drop columns:\n {drop_cols}')
    df.drop(drop_cols, axis=1, inplace=True)

    # fill nans
    df['last_reference_relative_loc'] = df['last_reference_relative_loc'].fillna(0)
    df['imp_changed'] = df['imp_changed'].fillna(-1)
    # rank_cols = [i for i in df.columns if 'rank' in i]
    # df[rank_cols] = df[rank_cols].fillna(0)
    # fill all nan with zeros now
    df.fillna(0, inplace=True)

    # some transformation
    to_log_median_cols = ['last_duration', 'session_duration', 'session_size', 'n_imps',
                          'mean_price', 'median_price', 'std_price', 'n_cfs',
                          'step', 'step_no_gap']
    prev_cols = [i for i in df.columns if 'prev' in i]
    to_log_median_cols.extend(prev_cols)

    logger.info(f'Performing log_median columns on:\n{np.array(to_log_median_cols)}')
    logger.warning('THIS PROBABLY WILL MAKE VALIDATION PERFORMANCE LOOK BETTER THAT IT ACTUALLY IS!!!')
    for col in tqdm(to_log_median_cols):
        log_median(df, col)

    # price_bins_cols = [i for i in df.columns if 'bin' in i]
    # df[price_bins_cols] = df[price_bins_cols]/5

    # specify columns for ohe-hot-encoding
    ohe_cols = ['last_action_type']

    for col in ohe_cols:
        logger.info(f'One hot: {col}')
        df[col] = df[col].astype(int)
        n_unique = df[col].nunique()
        df[col] = df[col].apply(lambda v: np.eye(n_unique, dtype=int)[v][1:])
        expand(df, col)

    logger.info(f'COLUMNS:\n{list(df.columns)}')

    filename = 'train_inputs_nn.snappy'
    df.to_parquet(os.path.join(Filepath.cache_path, filename), index=False)
    return df


def get_features_from_nn(nn_model, data, batch_size=1024):
    extract = Model(input=nn_model.inputs, outputs=nn_model.layers[-2].output)  # Dense(128,...)
    features = extract.predict(data, batch_size=batch_size)
    return features


def train(train_df, params, only_last=False, retrain=False):
    # path to where model is saved
    model_path = Filepath.model_path

    # if only use the last row of train_df to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_df = train_df.groupby('session_id').last().reset_index(drop=False)

    # grab unique session ids and use this to split, so that train_df with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_df['session_id'].unique()

    kf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=RS)

    # a bit prep-rocessing
    train_df = nn_prep(train_df)

    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    # record classifiers and mrr each training
    clfs = []
    mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(kf.split(unique_session_ids)):
        logger.info(f'Training fold {fold}: train len={len(trn_ind):,} | val len={len(val_ind):,}')
        # get session_id used for train
        trn_ids = unique_session_ids[trn_ind]
        trn_mask = train_df['session_id'].isin(trn_ids)

        x_trn, x_val = (train_df[trn_mask].reset_index(drop=True),
                        train_df[~trn_mask].reset_index(drop=True))

        # for validation only last row is needed
        x_val = x_val.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn, y_val = x_trn['target'].values, x_val['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        x_val.drop(['session_id', 'target'], axis=1, inplace=True)

        # data generator
        # train_gen = iterate_mini-batches(trn_num, trn_price, trn_click, y_trn, batch_size, shuffle=True)
        y_trn_binary = to_categorical(y_trn)
        y_val_binary = to_categorical(y_val)
        train_gen = iterate_minibatches(x_trn.values, y_trn_binary, batch_size, shuffle=True)
        val_gen = iterate_minibatches(x_val.values, y_val_binary, batch_size, shuffle=False)

        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'nn_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f"Loading model from existing '{model_filename}'")
            model = load_model(model_filename)
        else:
            # check if there is a trained model
            if os.path.isfile(model_filename):
                logger.info(f"Loading model from existing '{model_filename}'")
                model = load_model(model_filename)
            else:
                model = build_model(input_dim=x_trn.shape[1])
                nparams = model.count_params()
                opt = optimizers.Adam(lr=params['learning_rate'])
                # model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
                # model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
                model.compile(optimizer=opt, loss=custom_objective, metrics=['accuracy'])

                logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
                             f'| number of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
                logger.info(f'{model.summary()}')
                plot_model(model, to_file='./models/model.png')
                # add some callbacks
                callbacks = [ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, verbose=1)]
                log_dir = Filepath.tf_logs
                log_filename = ('{0}-batchsize{1}_epochs{2}_nparams_{3}'
                                .format(dt.now().strftime('%m-%d-%H-%M'), batch_size, n_epochs, nparams))
                # tb = TensorBoard(log_dir=os.path.join(log_dir, log_filename), write_graph=True, histogram_freq=20,
                #                  write_grads=True)
                tb = TensorBoard(log_dir=os.path.join(log_dir, log_filename), write_graph=True, write_grads=True)
                callbacks.append(tb)
                # lr
                # lr = LRTensorBoard(log_dir)
                # callbacks.append(lr)
                # logging
                # log = LoggingCallback(logger.info)
                # callbacks.append(log)
                if params['early_stop']:
                    # simple early stopping
                    es = EarlyStopping(monitor='val_loss', mode='min', patience=params['early_stopping_patience'],
                                       verbose=1)
                    callbacks.append(es)
                if params['reduce_on_plateau']:
                    # rp
                    rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=params['reduce_on_plateau_patience'],
                                           verbose=1)
                    callbacks.append(rp)
                if params['use_cyc']:
                    # step_size = (2 - 8)x(training iterations in epoch)
                    step_size = 8 * (len(y_trn) // batch_size)
                    logger.info(f'Using cyclic learning rate with step_size: {step_size}')
                    # clr = CyclicLR(base_lr=params['min_lr'], max_lr=params['max_lr'], step_size=step_size)
                    clr = CyclicLR(base_lr=params['min_lr'], max_lr=params['max_lr'], step_size=step_size,
                                   mode='exp_range', gamma=0.99994)
                    callbacks.append(clr)

                _ = model.fit_generator(train_gen,
                                        steps_per_epoch=len(y_trn) // batch_size,
                                        epochs=n_epochs,
                                        verbose=1,
                                        callbacks=callbacks,
                                        # validation_data=([val_num, val_price, val_click], y_val),
                                        # validation_data=(x_val.values, y_val),
                                        validation_data=val_gen,
                                        validation_steps=len(y_val) // batch_size)

            logger.info('Done training nn, now grabbing features from nn output to feed into lgb')
            x_trn_input = get_features_from_nn(model, x_trn)
            x_val_input = get_features_from_nn(model, x_val)

            logger.info(f'Input x from nn output shape: {x_trn_input.shape}')
            # dtrain = xgb.DMatrix(x_trn_input, label=y_trn)
            # dval = xgb.DMatrix(get_features_from_nn(model, x_val), label=y_val)
            lgb_trn_data = lgb.Dataset(x_trn_input, label=y_trn, free_raw_data=False)
            lgb_val_data = lgb.Dataset(x_val_input, label=y_val, free_raw_data=False)

            # train model
            logger.info('Starts training')
            train_params = {'boosting': 'gbdt',  # gbdt, dart, goss
                           'num_boost_round': 500,
                           'learning_rate': 0.02,
                           'early_stopping_rounds': 100,
                           'num_class': 25,
                           'objective': 'multiclass',
                           'metric': ['multi_logloss'],
                           'verbose': -1,
                           'seed': 42
                           }

            booster = lgb.train(train_params,
                                lgb_trn_data,
                                valid_sets=[lgb_trn_data, lgb_val_data],
                                valid_names=['train', 'val'],
                                feval=lgb_mrr,
                                # init_model=lgb.Booster(model_file=model_filename),
                                verbose_eval=100)

        booster.save_model(model_filename)

        # make prediction
        # x_trn = train_df[trn_mask].reset_index(drop=True)
        # x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        # y_trn = x_trn['target'].values
        # x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        # dtrain = xgb.DMatrix(x_trn, label=y_trn)

        trn_pred = booster.predict(x_trn_input)

        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = booster.predict(x_val_input)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(booster)
        mrrs.append((trn_mrr, val_mrr))

        # logger.info(f'Done training {fold}, took: {(time.time() - t1) / 60:.2f} mins')

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


if __name__ == '__main__':
    setup = {'nrows': 5000000,
             'recompute_train': False,
             'add_test': False,
             'only_last': False,
             'retrain': True,
             'recompute_test': False}

    params = {'batch_size': 512,
              'n_epochs': 250,
              'early_stopping_patience': 100,
              'reduce_on_plateau_patience': 30,
              'learning_rate': 0.001,
              'max_lr': 0.005,
              'min_lr': 0.0001,
              'use_cyc': False,
              'early_stop': False,
              'reduce_on_plateau': False
              }

    logger.info(f"\nSetup\n{'='*20}\n{pprint.pformat(setup)}\n{'='*20}")
    logger.info(f"\nParams\n{'='*20}\n{pprint.pformat(params)}\n{'='*20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], recompute=setup['recompute_train'])
    # train the model
    models, mrrs = train(train_inputs, params=params, only_last=setup['only_last'], retrain=setup['retrain'])
    mean_train_mrr = np.mean([mrr[0] for mrr in mrrs])
    mean_val_mrr = np.mean([mrr[1] for mrr in mrrs])
    # get the test inputs
    test_inputs = create_model_inputs(mode='test', recompute=setup['recompute_test'])
    # a bit preprocessing
    nn_prep(test_inputs)

    # make predictions on test
    logger.info('Load test sub csv')
    test_sub = pd.read_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'))
    test_sub = test_sub.groupby('session_id').last().reset_index(drop=False)
    test_sub.loc[:, 'impressions'] = test_sub.loc[:, 'impressions'].apply(lambda x: literal_eval(x))

    sub_popular = pd.read_csv(os.path.join(Filepath.data_path, 'submission_popular.csv'))
    sub_columns = sub_popular.columns

    # filter away the 0 padding and join list recs to string
    def create_recs(recs):
        return ' '.join([str(i) for i in recs if i != 0])

    test_predictions = []
    for c, clf in enumerate(models):
        logger.info(f'Generating predictions from model {c}')
        test_pred = clf.predict_proba(test_inputs)
        np.save(os.path.join(Filepath.sub_path, f'nn_test_{c}_pred.npy'), test_pred)
        test_predictions.append(test_pred)

    logger.info('Generating submission by averaging cv predictions')
    test_predictions = np.array(test_predictions).mean(axis=0)
    test_pred_label = np.argsort(test_predictions)[:, ::-1]
    np.save(os.path.join(Filepath.sub_path, f'test_pred_label.npy'), test_pred_label)

    test_impressions = np.array(list(test_sub['impressions'].values))
    test_impressions_pred = test_impressions[np.arange(len(test_impressions))[:, None], test_pred_label]
    test_sub.loc[:, 'recommendations'] = [create_recs(i) for i in test_impressions_pred]
    del test_sub['impressions']

    logger.info(f'Before merging: {test_sub.shape}')
    test_sub = pd.merge(test_sub, sub_popular, on='session_id')
    logger.info(f'After merging: {test_sub.shape}')
    del test_sub['item_recommendations']
    test_sub.rename(columns={'recommendations': 'item_recommendations'}, inplace=True)
    test_sub = test_sub[sub_columns]
    current_time = dt.now().strftime('%m-%d-%H-%M')
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'nn_sub_{current_time}_{mean_train_mrr:.4f}_{mean_val_mrr:.4f}.csv'),
                    index=False)
    logger.info('Done all')


