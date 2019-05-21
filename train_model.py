import os
import time
import numpy as np
from datetime import datetime as dt

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from keras.models import load_model
from keras.callbacks import Callback
from utils import get_logger, get_data_path
from model import build_model
from create_model_inputs import create_model_inputs
from plots import plot_hist, confusion_matrix

logger = get_logger('train_model')
Filepath = get_data_path()

TO_DO = ('1) maybe fillna -1 is too overwhelming for last_reference_id_index if normalized \n'
         '2) session_id size and dwell_time prior last click maybe need normalization in scale or maybe add batchnorm')

logger.info(TO_DO)


class LoggingCallback(Callback):
    """Callback that logs message at end of epoch.
    """
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
        self.print_fcn(msg)


# def iterate_minibatches(numerics, impressions, prices, cfilters, targets, batch_size, shuffle=True):
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
#             impressions_batch = impressions[excerpt]
#             prices_batch = prices[excerpt]
#             cfilters_batch = cfilters[excerpt]
#             targets_batch = targets[excerpt]
#
#             prices_batch = np.array([i.reshape(-1, 1) for i in prices_batch])
#             yield ([numerics_batch, impressions_batch, prices_batch,
#                     cfilters_batch], targets_batch)

def iterate_minibatches(numerics, prices, cfilters, targets, batch_size, shuffle=True):
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

            numerics_batch = numerics[excerpt]
            prices_batch = prices[excerpt]
            cfilters_batch = cfilters[excerpt]
            targets_batch = targets[excerpt]

            prices_batch = np.array([i.reshape(-1, 1) for i in prices_batch])
            yield ([numerics_batch, prices_batch,
                    cfilters_batch], targets_batch)


def train(train_inputs, params, retrain=False):
    cache_path = Filepath.cache_path
    model_path = Filepath.model_path

    # grab some info on n_cfs, this is used to create the filters ohe
    n_cfs = len(np.load(os.path.join(cache_path, 'filters_mapping.npy')).item())
    logger.info(f'Number of unique current_filters is: {n_cfs}')

    batch_size = params['batch_size']
    n_epochs = params['n_epochs']

    skf = StratifiedKFold(n_splits=6)
    models = []
    report = {}
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(skf.split(train_inputs['targets'], train_inputs['targets'])):
        logger.info(f'Training fold {fold}')
        report_fold = {}
        trn_numerics, val_numerics = train_inputs['numerics'][trn_ind], train_inputs['numerics'][val_ind]
        # trn_imp, val_imp = train_inputs['impressions'][trn_ind], train_inputs['impressions'][val_ind]
        trn_price, val_price = train_inputs['prices'][trn_ind], train_inputs['prices'][val_ind]
        trn_cfilter, val_cfilter = train_inputs['cfilters'][trn_ind], train_inputs['cfilters'][val_ind]
        y_trn, y_val = train_inputs['targets'][trn_ind], train_inputs['targets'][val_ind]
        report_fold['train_len'] = len(y_trn)
        report_fold['val_len'] = len(y_val)
        # data generator
        # train_gen = iterate_minibatches(trn_numerics, trn_imp, trn_price, trn_cfilter, y_trn, batch_size, shuffle=True)
        # val_gen = iterate_minibatches(val_numerics, val_imp, val_price, val_cfilter, y_val, batch_size, shuffle=False)
        train_gen = iterate_minibatches(trn_numerics, trn_price, trn_cfilter, y_trn, batch_size, shuffle=True)
        val_gen = iterate_minibatches(val_numerics, val_price, val_cfilter, y_val, batch_size, shuffle=False)


        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f'Loading model from existing {model_filename}')
            model = load_model(model_filename)
        else:
            model = build_model(n_cfs, params=params)

            # print out model info
            nparams = model.count_params()
            report['nparams'] = nparams
            logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
                         f'| number of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
            logger.info(f'{model.summary()}')
            plot_model(model, to_file='./models/model.png')
            # add some callbacks
            callbacks = [ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True, verbose=1)]
            log_dir = Filepath.tf_logs
            log_filename = ('{0}-batchsize{1}_epochs{2}_tcn_filter{3}_fsize{4}_ns{5}_ldial{6}_nparams_{7}'
                            .format(dt.now().strftime('%m-%d-%H-%M'), batch_size, n_epochs,
                                    params['tcn_params']['nb_filters'], params['tcn_params']['kernel_size'],
                                    params['tcn_params']['nb_stacks'], params['tcn_params']['dilations'][-1],
                                    nparams))
            tb = TensorBoard(log_dir=os.path.join(log_dir, log_filename), write_graph=True, write_grads=True)
            callbacks.append(tb)
            # simple early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', patience=params['early_stopping'], verbose=1)
            callbacks.append(es)
            # rp
            rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=params['reduce_on_plateau'], verbose=1)
            callbacks.append(rp)
            # logging
            log = LoggingCallback(logger.info)
            callbacks.append(log)

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=len(y_trn) // batch_size,
                                          epochs=n_epochs,
                                          verbose=1,
                                          callbacks=callbacks,
                                          validation_data=val_gen,
                                          validation_steps=len(y_val) // batch_size)

        # # make prediction
        # trn_pred = model.predict(x=[trn_numerics, trn_imp, trn_price[:, :, None], trn_cfilter], batch_size=1024)
        # trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        # plot_hist(trn_pred_label, y_trn, 'train')
        # confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        # trn_mrr = np.mean(1 / (trn_pred_label + 1))
        #
        # val_pred = model.predict(x=[val_numerics, val_imp, val_price[:, :, None], val_cfilter], batch_size=1024)
        # val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        # plot_hist(val_pred_label, y_val, 'validation')
        # confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        # val_mrr = np.mean(1 / (val_pred_label + 1))
        # logger.info(f'train mrr: {trn_mrr:.2f} | val mrr: {val_mrr:.2f}')

        # make prediction
        trn_pred = model.predict(x=[trn_numerics, trn_price[:, :, None], trn_cfilter], batch_size=1024)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = model.predict(x=[val_numerics, val_price[:, :, None], val_cfilter], batch_size=1024)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.2f} | val mrr: {val_mrr:.2f}')

        models.append(model)

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return models


if __name__ == '__main__':
    setup = {'nrows': 5000000,
             'recompute_train': False,
             'retrain': True,
             'recompute_test': True}

    params = {'batch_size': 256,
              'n_epochs': 500,
              'early_stopping': 50,
              'reduce_on_plateau': 30,
              'tcn_params':
                  {'nb_filters': 32,
                   'kernel_size': 3,
                   'nb_stacks': 2,
                   'padding': 'causal',
                   'dilations': [1, 2, 4],
                   'use_skip_connections': True,
                   'dropout_rate': 0.2,
                   'return_sequences': False,
                   'name': 'tcn'},
              'learning_rate': 0.001,
              }

    logger.info(f"\nSetup\n{'='*20}\n{setup}\n{'='*20}")
    logger.info(f"\nParams\n{'='*20}\n{params}\n{'='*20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], recompute=setup['recompute_train'])
    # train the model
    models = train(train_inputs, params=params, retrain=setup['retrain'])
    # get the test inputs
    test_inputs = create_model_inputs(mode='test', nrows=setup['nrows'], recompute=setup['recompute_test'])
    # make predictions on test
    logger.info('Load test sub csv')
    test_sub = pd.read_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'))
    sub_popular = pd.read_csv(os.path.join(Filepath.data_path, 'submission_popular.csv'))
    sub_columns = sub_popular.columns

    # filter away the 0 padding and join list recs to string
    def create_recs(recs):
        return ' '.join([i for i in recs if i != 0])


    numerics, impressions, prices, cfilters = test_inputs['numerics'], test_inputs['impressions'], \
                                              test_inputs['prices'], test_inputs['cfilters'],
    test_predictions = []
    for m, model in enumerate(models):
        # test_sub_m = test_sub.copy()
        logger.info(f'Generating predictions from model {m}')
        test_pred = model.predict(x=[numerics, impressions, prices[:, :, None], cfilters], batch_size=1024)
        test_predictions.append(test_pred)

    logger.info('Generating submission by averaging cv predictions')
    test_predictions = np.array(test_predictions).mean(axis=0)
    test_pred_label = np.argsort(test_predictions)[:, ::-1]
    np.save(os.path.join(Filepath.sub_path, f'test_pred_label.npy'), test_pred_label)

    # pad to 25
    test_sub['impressions'] = test_sub['impressions'].str.split('|')
    print(test_sub['impressions'].str.len().describe())
    test_sub['impressions'] = test_sub['impressions'].apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))
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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'sub.csv'), index=False)
    logger.info('Done all')

