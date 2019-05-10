import os
import time
import numpy as np
from datetime import datetime as dt

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from keras.models import load_model

from utils import get_logger, check_dir
from nn_model import build_model
from create_nn_train_input import create_train_inputs
from create_nn_test_input import create_test_inputs
from plots import plot_hist, confusion_matrix

logger = get_logger('train_nn')

TO_DO = ('1) maybe fillna -1 is too overwhelming for last_reference_id_index if normalized \n'
         '2) session_id size and dwell_time prior last click maybe need normalization in scale or maybe add batchnorm')

logger.info(TO_DO)


# def iterate_minibatches(numerics, impressions, prices, cfilters, targets,
#                         batch_size, shuffle=True):
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


def get_steps(seq, batch_size):
    sizes = np.array([len(i) for i in seq])
    # unique_sizes = np.array(list(set(sizes)))
    size_ctn = pd.value_counts(sizes)
    # get remainder
    remainder = size_ctn % batch_size
    # the integer gives the complete cycle of number of batch_size,
    # if the remainder is not 0 then it means there will be one more
    return (size_ctn//batch_size + (remainder != 0).astype(int)).sum()


def iterate_minibatches(numerics, impressions, prices, cfilters, targets,
                        batch_size, shuffle=True):

    sizes = np.array([len(i) for i in prices])
    unique_sizes = np.array(list(set(sizes)))

    while True:
        if shuffle:
            np.random.shuffle(unique_sizes)

        # going over each sizes
        for s in unique_sizes:
            mask = sizes == s
            targets_s = targets[mask]
            # number of records in this bucket
            n_s = len(targets_s)
            indices = np.arange(n_s)
            if shuffle:
                np.random.shuffle(indices)

            remainder = n_s % batch_size
            for start_idx in range(0, n_s, batch_size):
                if remainder != 0 and start_idx + batch_size >= n_s:
                    excerpt = indices[n_s-batch_size:n_s]
                else:
                    excerpt = indices[start_idx:start_idx + batch_size]

                numerics_batch = numerics[mask][excerpt]
                impressions_batch = impressions[mask][excerpt]
                prices_batch = prices[mask][excerpt]
                cfilters_batch = cfilters[mask][excerpt]
                targets_batch = targets[mask][excerpt]

                prices_batch = np.array([i.reshape(-1, 1) for i in prices_batch])
                yield ([numerics_batch, impressions_batch, prices_batch,
                        cfilters_batch], targets_batch)


def train(numerics, impressions, prices, cfilters, targets, params, retrain=False):
    # grab some info on n_cfs, this is used to create the filters ohe
    n_cfs = len(np.load('./cache/filters_mapping.npy').item())
    logger.info(f'Number of unique current_filters is: {n_cfs}')

    batch_size = params['batch_size']
    n_epochs = params['n_epochs']
    model_params = params['model_params']

    skf = StratifiedKFold(n_splits=6)
    models = []
    report = {}
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(skf.split(targets, targets)):
        report_fold = {}
        trn_numerics, val_numerics = numerics[trn_ind], numerics[val_ind]
        trn_imp, val_imp = impressions[trn_ind], impressions[val_ind]
        trn_price, val_price = prices[trn_ind], prices[val_ind]
        trn_cfilter, val_cfilter = cfilters[trn_ind], cfilters[val_ind]
        y_trn, y_val = targets[trn_ind], targets[val_ind]
        report_fold['train_len'] = len(y_trn)
        report_fold['val_len'] = len(y_val)
        # data generator
        train_gen = iterate_minibatches(trn_numerics, trn_imp, trn_price, trn_cfilter, y_trn,
                                        batch_size, shuffle=True)

        val_gen = iterate_minibatches(val_numerics, val_imp, val_price, val_cfilter, y_val,
                                      batch_size, shuffle=False)

        # =====================================================================================
        # create model
        model_filename = f'./models/cv{fold}.model'
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f'Loading model from existing {model_filename}')
            model = load_model(model_filename)
        else:
            model = build_model(n_cfs)

            # print out model info
            nparams = model.count_params()
            report['nparams'] = nparams
            logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
                         f'| number of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
            logger.info(f'{model.summary()}')
            check_dir('./models')
            plot_model(model, to_file='./models/model.png')
            # add some callbacks
            callbacks = [ModelCheckpoint(model_filename, save_best_only=True, verbose=1)]
            log_dir = './logs/{}'.format(dt.now().strftime('%m-%d-%H-%M'))
            tb = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=True)
            callbacks.append(tb)
            # simple early stopping
            es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
            callbacks.append(es)
            # rp
            rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=1)
            callbacks.append(rp)

            history = model.fit_generator(train_gen,
                                          steps_per_epoch=len(y_trn) // batch_size,
                                          epochs=n_epochs,
                                          verbose=1,
                                          callbacks=callbacks,
                                          validation_data=val_gen,
                                          validation_steps=len(y_val) // batch_size)

        # make prediction
        trn_pred = model.predict(x=[trn_numerics, trn_imp, trn_price[:, :, None], trn_cfilter], batch_size=1024)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)

        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = model.predict(x=[val_numerics, val_imp, val_price[:, :, None], val_cfilter], batch_size=1024)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.2f} | val mrr: {val_mrr:.2f}')

        models.append(model)
        break
    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return models


if __name__ == '__main__':
    setup = {'nrows': 5000000,
             'recompute_train': False,
             'retrain': True,
             'recompute_test': False}
    params = {'batch_size': 128,
              'n_epochs': 100,
              'model_params': None}
    # logger.info(pprint.pformat(setup))
    # logger.info(pprint.pformat(params))
    logger.info(f"Setup\n{'='*20}\n{setup}\n{'='*20}")
    logger.info(f"Params\n{'='*20}\n{params}\n{'='*20}")

    # first create training inputs
    numerics, impressions, prices, cfilters, targets = create_train_inputs(nrows=setup['nrows'],
                                                                           recompute=setup['recompute_train'])
    # train the model
    models = train(numerics, impressions, prices, cfilters, targets, params=params, retrain=setup['retrain'])
    # get the test inputs
    numerics, impressions, prices, cfilters = create_test_inputs(recompute=setup['recompute_test'])
    # make predictions on test
    check_dir('./subs')
    logger.info('Load test sub csv')
    test_sub = pd.read_csv('./cache/test_sub.csv')
    sub_popular = pd.read_csv('../data/submission_popular.csv')
    sub_columns = sub_popular.columns

    # filter away the 0 padding and join list recs to string
    def create_recs(recs):
        return ' '.join([i for i in recs if i != 0])

    for m, model in enumerate(models):
        test_sub_m = test_sub.copy()

        logger.info(f'Generating submission from model {m}')
        test_pred = model.predict(x=[numerics, impressions, prices[:, :, None], cfilters], batch_size=1024)
        test_pred_label = np.argsort(test_pred)[:, ::-1]
        np.save(f'./models/test_pred_label{m}.npy', test_pred_label)

        # pad to 25
        test_sub_m['impressions'] = test_sub_m['impressions'].str.split('|')
        print(test_sub_m['impressions'].str.len().describe())
        test_sub_m['impressions'] = test_sub_m['impressions'].apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))
        test_impressions = np.array(list(test_sub_m['impressions'].values))

        test_impressions_pred = test_impressions[np.arange(len(test_impressions))[:, None], test_pred_label]
        test_sub_m.loc[:, 'recommendations'] = [create_recs(i) for i in test_impressions_pred]
        del test_sub_m['impressions']

        logger.info(f'Before merging: {test_sub_m.shape}')
        test_sub_m = pd.merge(test_sub_m, sub_popular, on='session_id')
        logger.info(f'After merging: {test_sub_m.shape}')
        del test_sub_m['item_recommendations']
        test_sub_m.rename(columns={'recommendations': 'item_recommendations'}, inplace=True)
        test_sub_m = test_sub_m[sub_columns]
        test_sub_m.to_csv(f'./subs/sub_{m}.csv', index=False)
    logger.info('Done all')

