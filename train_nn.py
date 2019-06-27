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
import tensorflow as tf

from model import build_model, multimodal_model
from clr_callback import CyclicLR
from create_model_inputs import create_model_inputs, expand
from utils import get_logger, get_data_path
from plots import plot_hist, confusion_matrix


logger = get_logger('train_nn')
Filepath = get_data_path()
RS = 42


def bpr_max_loss(y_true, y_pred):
    l2 = True
    weight = 1.
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


def iterate_minibatches(input_price, input_prev, input_star, input_rating, input_rest, targets, batch_size, shuffle=True):
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

            batch_price = input_price[excerpt]
            batch_prev = input_prev[excerpt]
            batch_star = input_star[excerpt]
            batch_rating = input_rating[excerpt]
            batch_rest = input_rest[excerpt]

            targets_batch = targets[excerpt]
            yield ([batch_price, batch_prev, batch_star, batch_rating, batch_rest], targets_batch)


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

    # a bit prep-processing
    train_df = nn_prep(train_df)

    # get different modal
    price_cols = [i for i in train_df.columns if 'price' in i]
    prev_cols = [i for i in train_df if 'prev' in i]
    star_cols = [i for i in train_df if 'star' in i]
    rating_cols = [i for i in train_df if 'rating' in i]
    rest_cols = [i for i in train_df if (i not in price_cols + prev_cols + star_cols + rating_cols)
                 and i not in ['session_id', 'target']]
    logger.info(f'Multimodal: price, prev, star, rating columns and the rest:\n{rest_cols}')

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
        y_trn_binary = to_categorical(y_trn)
        y_val_binary = to_categorical(y_val)
        # train_gen = iterate_minibatches(x_trn.values, y_trn_binary, batch_size, shuffle=True)
        # val_gen = iterate_minibatches(x_val.values, y_val_binary, batch_size, shuffle=False)

        train_gen = iterate_minibatches(x_trn[price_cols].values,
                                        x_trn[prev_cols].values,
                                        x_trn[star_cols].values,
                                        x_trn[rating_cols].values,
                                        x_trn[rest_cols].values,
                                        y_trn_binary, batch_size, shuffle=True)

        val_gen = iterate_minibatches(x_val[price_cols].values,
                                      x_val[prev_cols].values,
                                      x_val[star_cols].values,
                                      x_val[rating_cols].values,
                                      x_val[rest_cols].values,
                                      y_val_binary, batch_size, shuffle=True)

        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'nn_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f"Loading model from existing '{model_filename}'")
            model = load_model(model_filename)
        else:
            # model = build_model(input_dim=x_trn.shape[1])
            model = multimodal_model(price_dim=len(price_cols), prev_dim=len(prev_cols), star_dim=len(star_cols),
                                     rating_dim=len(rating_cols), rest_dim=len(rest_cols))
            nparams = model.count_params()
            opt = optimizers.Adam(lr=params['learning_rate'])
            # model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
            # model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
            model.compile(optimizer=opt, loss=bpr_max_loss, metrics=['accuracy'])

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

        # make prediction
        x_trn = train_df[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        y_trn = x_trn['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)

        trn_pred = model.predict(x=x_trn.values, batch_size=1024)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        # plot_hist(trn_pred_label, y_trn, 'train')
        # confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))
        np.save(os.path.join(Filepath.sub_path, f'nn_trn_{fold}_pred.npy'), trn_pred)

        val_pred = model.predict(x=x_val.values, batch_size=1024)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        # plot_hist(val_pred_label, y_val, 'validation')
        # confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')
        np.save(os.path.join(Filepath.sub_path, f'nn_val_{fold}_pred.npy'), val_pred)

        clfs.append(model)
        mrrs.append((trn_mrr, val_mrr))

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
              'n_epochs': 500,
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


