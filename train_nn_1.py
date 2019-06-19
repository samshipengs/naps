import os
import re
import time
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ast import literal_eval

from sklearn.model_selection import KFold
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from keras.models import load_model
from model import build_model
from clr_callback import CyclicLR
from create_model_inputs import create_model_inputs
from utils import get_logger, get_data_path
from plots import plot_hist, confusion_matrix


logger = get_logger('train_model')
Filepath = get_data_path()
RS = 42


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


# def iterate_minibatches(numerics, prices, clicks, targets, batch_size, shuffle=True):
#     # default we will shuffle
#     indices = np.arange(len(targets))
#     while True:
#         if shuffle:
#             np.random.shuffle(indices)
#
#         remainder = len(targets) % batch_size
#         for start_idx in range(0, len(targets), batch_size):
#             if remainder != 0 and start_idx + batch_size >= len(targets):
#                 excerpt = indices[len(targets) - batch_size:len(targets)]
#             else:
#                 excerpt = indices[start_idx:start_idx + batch_size]
#
#             numerics_batch = numerics[excerpt]
#             prices_batch = prices[excerpt]
#             clicks_batch = clicks[excerpt]
#             targets_batch = targets[excerpt]
#
#             yield ([numerics_batch, prices_batch, clicks_batch], targets_batch)

def iterate_minibatches(input_x, targets, embed_ind, num_ind, batch_size, shuffle=True):
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

            input_batch = input_x[:, num_ind][excerpt]
            emb = [input_x[:, i][excerpt] for i in embed_ind]
            # prices_batch = prices[excerpt]
            # clicks_batch = clicks[excerpt]
            targets_batch = targets[excerpt]

            # yield ([numerics_batch, prices_batch, clicks_batch], targets_batch)
            yield ([input_batch]+emb, targets_batch)


def log_median(df, col):
    df[col] = np.log1p(df[col])
    df[col] = df[col]/np.median(df[col])


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
    # if only use the last row of train_inputs to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_inputs = train_inputs.groupby('session_id').last().reset_index(drop=False)

    # grab unique session ids and use this to split, so that train_inputs with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_inputs['session_id'].unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=RS)

    # fill nans
    train_inputs['last_reference_relative_loc'] = train_inputs['last_reference_relative_loc'].fillna(-1)
    train_inputs['imp_changed'] = train_inputs['imp_changed'].fillna(-1)
    rank_cols = [i for i in train_inputs.columns if 'rank' in i]
    train_inputs[rank_cols] = train_inputs[rank_cols].fillna(-1)
    train_inputs.fillna(0, inplace=True)

    # some transformation
    log_median(train_inputs, 'last_duration')
    log_median(train_inputs, 'session_duration')
    # ohe
    embed_cols = ['last_action_type', 'fs', 'sort_order']
    for col in embed_cols:
        train_inputs[col] = train_inputs[col].astype(int)
        # train_inputs = pd.get_dummies(train_inputs, prefix=[col], drop_first=True)

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

        # for validation only last row is needed
        x_val = x_val.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn, y_val = x_trn['target'].values, x_val['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        x_val.drop(['session_id', 'target'], axis=1, inplace=True)

        # get embedding cols index
        embed_ind = [k for k, v in enumerate(x_trn.columns) if v in embed_cols]
        num_ind = [k for k, v in enumerate(x_trn.columns) if v not in embed_cols]

        # get each modal
        # trn_num, val_num = x_trn[numeric_cols].values, x_val[numeric_cols].values
        # trn_price, val_price = x_trn[price_cols].values, x_val[price_cols].values
        # trn_click, val_click = x_trn[click_cols].values, x_val[click_cols].values

        # data generator
        # train_gen = iterate_minibatches(trn_num, trn_price, trn_click, y_trn, batch_size, shuffle=True)
        train_gen = iterate_minibatches(x_trn.values, y_trn, embed_ind, num_ind, batch_size, shuffle=True)

        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'nn_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f"Loading model from existing '{model_filename}'")
            model = load_model(model_filename)
        else:
            model = build_model()
            nparams = model.count_params()

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
            lr = LRTensorBoard(log_dir)
            callbacks.append(lr)
            # logging
            log = LoggingCallback(logger.info)
            callbacks.append(log)
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

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=len(y_trn) // batch_size,
                                          epochs=n_epochs,
                                          verbose=1,
                                          callbacks=callbacks,
                                          # validation_data=([val_num, val_price, val_click], y_val),
                                          validation_data=([x_val.values[:, num_ind]] + [x_val.values[:, i] for i in embed_ind],
                                                           y_val),
                                          validation_steps=len(y_val) // batch_size)

        # make prediction
        x_trn = train_inputs[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        y_trn = x_trn['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)

        trn_pred = model.predict(x=x_trn.values, batch_size=1024)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = model.predict(x=x_val.values, batch_size=1024)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(model)
        mrrs.append((trn_mrr, val_mrr))

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


if __name__ == '__main__':
    setup = {'nrows': 5000000,
             'recompute_train': False,
             'add_test': False,
             'only_last': True,
             'retrain': True,
             'recompute_test': False}

    params = {'batch_size': 512,
              'n_epochs': 100,
              'early_stopping_patience': 50,
              'reduce_on_plateau_patience': 30,
              'learning_rate': 0.001,
              'max_lr': 0.005,
              'min_lr': 0.0001,
              'use_cyc': True,
              'early_stop': False,
              'reduce_on_plateau': False
              }

    logger.info(f"\nSetup\n{'='*20}\n{setup}\n{'='*20}")
    logger.info(f"\nParams\n{'='*20}\n{params}\n{'='*20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], recompute=setup['recompute_train'])
    # train the model
    models, mrrs = train(train_inputs, params=params, only_last=setup['only_last'], retrain=setup['retrain'])
    train_mrr = np.mean([mrr[0] for mrr in mrrs])
    val_mrr = np.mean([mrr[1] for mrr in mrrs])
    # get the test inputs
    test_inputs = create_model_inputs(mode='test', nrows=setup['test_rows'], recompute=setup['recompute_test'])
    # test_inputs = test_inputs.sort_values(by=['cust'])
    test_ids = np.load(os.path.join(Filepath.gbm_cache_path, 'test_ids.npy'))
    test_inputs['session_id'] = test_ids
    test_inputs = test_inputs.groupby('session_id').last().reset_index(drop=True)
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
        # test_sub_m = test_sub.copy()
        logger.info(f'Generating predictions from model {c}')
        test_pred = clf.predict_proba(test_inputs)
        test_predictions.append(test_pred)

    logger.info('Generating submission by averaging cv predictions')
    test_predictions = np.array(test_predictions).mean(axis=0)
    test_pred_label = np.argsort(test_predictions)[:, ::-1]
    np.save(os.path.join(Filepath.sub_path, f'test_pred_label.npy'), test_pred_label)

    # pad to 25
    # test_sub['impressions'] = test_sub['impressions'].str.split('|')
    # print(test_sub['impressions'].str.len().describe())
    # test_sub['impressions'] = test_sub['impressions'].apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))
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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'cat_sub_{current_time}.csv'), index=False)
    logger.info('Done all')
