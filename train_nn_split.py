import os
import re
import time
import pprint
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ast import literal_eval

from sklearn.model_selection import KFold, ShuffleSplit
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import optimizers
from keras.utils import plot_model
from keras.models import load_model
import tensorflow as tf

from model import build_model
from clr_callback import CyclicLR
from create_model_inputs import create_model_inputs, expand
from utils import get_logger, get_data_path
from plots import plot_hist, confusion_matrix


logger = get_logger('train_model')
Filepath = get_data_path()
RS = 42


# def custom_objective(y_true, y_pred):
#     pred_value = K.max(y_pred*y_true, axis=1)
#     diff = K.reshape(pred_value, (512, 1)) - y_pred
#     incorrect_pred_mask = diff < 0
#     # select the one that is incorrectly predicted
#     incorrects = tf.boolean_mask(-diff, incorrect_pred_mask)
#     return K.sum(K.flatten(incorrects))


# def custom_objective(y_true, y_pred):
#     pred_value = K.max(y_pred*y_true, axis=1)
#     diff = K.max(y_pred, axis=1) - pred_value
#     return K.sum(diff)


def iterate_minibatches(input_x, targets, batch_size, shuffle=True):
    # default we will shuffle
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
    df[col] = np.log1p(df[col])
    df[col] = df[col]/np.median(df[col])


def nn_prep(df):
    # fill nans
    df['last_reference_relative_loc'] = df['last_reference_relative_loc'].fillna(-1)
    df['imp_changed'] = df['imp_changed'].fillna(-1)
    rank_cols = [i for i in df.columns if 'rank' in i]
    df[rank_cols] = df[rank_cols].fillna(-1)
    df.fillna(0, inplace=True)

    # some transformation
    log_median(df, 'last_duration')
    log_median(df, 'session_duration')
    # ohe
    # ohe_cols = ['last_action_type', 'fs', 'sort_order']
    ohe_cols = ['last_action_type']

    for col in ohe_cols:
        logger.info(f'One hot: {col}')
        df[col] = df[col].astype(int)
        # df = pd.get_dummies(df, prefix=[col], drop_first=True)
        n_unique = df[col].nunique()
        df[col] = df[col].apply(lambda v: np.eye(n_unique, dtype=int)[v][1:])
        expand(df, col)


def train(train_inputs, params, only_last=False, retrain=False):
    # path to where model is saved
    model_path = Filepath.model_path

    # specify some columns that we do not want in training
    cf_cols = [i for i in train_inputs.columns if 'current_filters' in i]
    price_cols = [i for i in train_inputs.columns if re.match(r'prices_\d', i)]
    drop_cols = cf_cols + price_cols  # + ['country', 'platform']
    # drop cf col for now
    train_inputs.drop(drop_cols, axis=1, inplace=True)
    logger.debug(f'train columns: {train_inputs.columns}')

    # add sizes info
    train_inputs['length'] = train_inputs.groupby('session_id')['session_id'].transform('size')

    # if only use the last row of train_inputs to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_inputs = train_inputs.groupby('session_id').last().reset_index(drop=False)

    # grab unique session ids and use this to split, so that train_inputs with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_inputs['session_id'].unique()

    kf = ShuffleSplit(n_splits=5, test_size=0.15, random_state=RS)

    # a bit preprocessing
    nn_prep(train_inputs)

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
        trn_mask = train_inputs['session_id'].isin(trn_ids)

        x_trn, x_val = (train_inputs[trn_mask].reset_index(drop=True),
                        train_inputs[~trn_mask].reset_index(drop=True))

        # split the training into two models, first with only session_size=1 second with > 1
        trn_ones_mask = x_trn['length'] == 1
        val_ones_mask = x_val['length'] == 1
        n_ones_trn, n_ones_val = trn_ones_mask.sum(), val_ones_mask.sum()
        n_more_trn, n_more_val = (~trn_ones_mask).sum(), (~val_ones_mask).sum()
        logger.info(f'Train has {n_ones_trn:,} ones session and {n_more_trn:,} more | '
                    f'Val has {n_ones_val:,} ones session and {n_more_val:,} more')

        # train
        x_trn_ones, x_trn_more = (x_trn[trn_ones_mask].reset_index(drop=True),
                                  x_trn[~trn_ones_mask].reset_index(drop=True))
        # val
        x_val_ones, x_val_more = (x_val[val_ones_mask].reset_index(drop=True),
                                  x_val[~val_ones_mask].reset_index(drop=True))

        # for validation only last row is needed
        x_val = x_val.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn_ones, y_val_ones = x_trn_ones['target'].values, x_val_ones['target'].values
        y_trn_more, y_val_more = x_trn_more['target'].values, x_val_more['target'].values

        remove_ones_cols = [c for c in x_trn.columns if ('prev' in c) or ('last' in c)]
        remove_ones_cols += ['imp_changed', 'fs', 'sort_order']
        remove_ones_cols += ['session_id', 'length', 'target']
        x_trn_ones.drop(remove_ones_cols, axis=1, inplace=True)
        x_val_ones.drop(remove_ones_cols, axis=1, inplace=True)

        x_trn_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)
        x_val_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)

        # data generator
        train_gen_ones = iterate_minibatches(x_trn_ones.values, y_trn_ones, batch_size, shuffle=True)
        val_gen_ones = iterate_minibatches(x_val_ones.values, y_val_ones, batch_size, shuffle=False)
        train_gen_more = iterate_minibatches(x_trn_more.values, y_trn_more, batch_size, shuffle=True)
        val_gen_more = iterate_minibatches(x_val_more.values, y_val_more, batch_size, shuffle=False)

        # =====================================================================================
        # create model
        model_one_filename = os.path.join(model_path, f'nn_one_cv{fold}.model')
        model_more_filename = os.path.join(model_path, f'nn_more_cv{fold}.model')

        model_ones = build_model(input_dim=98)
        nparams_ones = model_ones.count_params()
        # opt = optimizers.Adam(lr=params['learning_rate'])
        opt = optimizers.Adagrad(lr=params['learning_rate'])
        model_ones.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        # logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
        #              f'| number of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
        callbacks = [ModelCheckpoint(model_one_filename, monitor='val_loss', save_best_only=True, verbose=1)]
        log_dir = Filepath.tf_logs
        log_filename = ('{0}-batchsize{1}_epochs{2}_nparams_{3}'
                        .format(dt.now().strftime('%m-%d-%H-%M'), batch_size, n_epochs, nparams_ones))
        tb = TensorBoard(log_dir=os.path.join(log_dir, log_filename), write_graph=True, write_grads=True)
        callbacks.append(tb)

        history_ones = model_ones.fit_generator(train_gen_ones,
                                                steps_per_epoch=len(y_trn_ones) // batch_size,
                                                epochs=n_epochs,
                                                verbose=1,
                                                callbacks=callbacks,
                                                validation_data=val_gen_ones,
                                                validation_steps=len(y_val_ones) // batch_size)
        # make prediction
        x_trn = train_inputs[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        # train
        trn_ones_mask = x_trn['length'] == 1
        x_trn_ones, x_trn_more = (x_trn[trn_ones_mask].reset_index(drop=True),
                                  x_trn[~trn_ones_mask].reset_index(drop=True))

        y_trn_ones, y_trn_more = x_trn_ones['target'].values, x_trn_more['target'].values
        x_trn_ones.drop(remove_ones_cols, axis=1, inplace=True)
        x_trn_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)
        x_trn_pred_ones = model_ones.predict(x=x_trn_ones.values, batch_size=1024)
        x_val_pred_ones = model_ones.predict(x=x_val_ones.values, batch_size=1024)

        model_more = build_model(input_dim=163)
        nparams_more = model_ones.count_params()
        # opt = optimizers.Adam(lr=params['learning_rate'])
        opt = optimizers.Adagrad(lr=params['learning_rate'])
        model_more.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        # logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
        #              f'| number of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
        callbacks = [ModelCheckpoint(model_more_filename, monitor='val_loss', save_best_only=True, verbose=1)]
        log_dir = Filepath.tf_logs
        log_filename = ('{0}-batchsize{1}_epochs{2}_nparams_{3}'
                        .format(dt.now().strftime('%m-%d-%H-%M'), batch_size, n_epochs, nparams_more))
        tb = TensorBoard(log_dir=os.path.join(log_dir, log_filename), write_graph=True, write_grads=True)
        callbacks.append(tb)

        history_more = model_more.fit_generator(train_gen_more,
                                                steps_per_epoch=len(y_trn_more) // batch_size,
                                                epochs=n_epochs,
                                                verbose=1,
                                                callbacks=callbacks,
                                                validation_data=val_gen_more,
                                                validation_steps=len(y_val_more) // batch_size)
        x_trn_pred_more = model_more.predict(x=x_trn_more.values, batch_size=1024)
        x_val_pred_more = model_more.predict(x=x_val_more.values, batch_size=1024)

        print(x_trn_ones.shape, x_trn_more.shape)
        trn_pred = np.concatenate((x_trn_pred_ones, x_trn_pred_more), axis=0)
        y_trn = np.concatenate((y_trn_ones, y_trn_more), axis=0)
        print(trn_pred.shape, y_trn.shape)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        print(trn_pred_label.shape)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))
        # save prediction
        # np.save(os.path.join(model_path, 'cat_trn_0_pred.npy'), trn_pred)

        val_pred = np.concatenate((x_val_pred_ones, x_val_pred_more), axis=0)
        y_val = np.concatenate((y_val_ones, y_val_more), axis=0)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        np.save(os.path.join(model_path, 'lgb_val_0_pred.npy'), val_pred)

        clfs.append(clf)
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
              'n_epochs': 50,
              'early_stopping_patience': 100,
              'reduce_on_plateau_patience': 30,
              'learning_rate': 0.01,
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
    train_mrr = np.mean([mrr[0] for mrr in mrrs])
    val_mrr = np.mean([mrr[1] for mrr in mrrs])
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
        np.save(os.path.join(Filepath.sub_path, 'nn_test_0_pred.npy'), test_pred)
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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'nn_sub_{current_time}_{train_mrr:.4f}_{val_mrr:.4f}.csv'),
                    index=False)
    logger.info('Done all')


