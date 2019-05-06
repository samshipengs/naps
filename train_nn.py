import pandas as pd
import numpy as np
import datetime, time, os, gc, re, sys
from functools import partial
import matplotlib.pyplot as plt
from utils import ignore_warnings, load_data, get_logger, check_dir
from clean_session import preprocess_sessions
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import StratifiedKFold
from datetime import datetime as dt
from nn_model_simple import build_model
from keras.utils import plot_model
from create_nn_input import create_inputs


logger = get_logger('train_nn')


def iterate_minibatches(numerics, impressions, prices, cfilters, targets,
                        batch_size, shuffle=True):
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
            impressions_batch = impressions[excerpt]  # .reshape(batch_size, -1, 157)
            prices_batch = prices[excerpt]  # .reshape(batch_size, -1, 1)
            cfilters_batch = cfilters[excerpt]
            targets_batch = targets[excerpt]

            prices_batch = np.array([i.reshape(-1, 1) for i in prices_batch])
            yield ([numerics_batch, impressions_batch, prices_batch,
                    cfilters_batch], targets_batch)


def get_data(nrows=100000):
    train = create_inputs(nrows)

    logger.info('Grabbing list of inputs')
    logger.info('Normalizing price')
    # maybe normalize to percentage within each records, check does each item_id have the same price over all records
    def normalize(ps):
        p_arr = np.array(ps)
        return p_arr / (p_arr.max())

    train['prices'] = train['prices'].apply(normalize)
    # PRICES
    prices = np.array(list(train['prices'].values))
    del train['prices']

    logger.info('Getting impressions')
    # IMPRESSIONS
    impressions = np.array(list(train['impressions'].values))
    del train['impressions']

    logger.info('Getting current_filters')
    # CURRENT_FILTERS
    cfilters = np.array(list(train['cfs'].values))
    del train['cfs']

    logger.info('Getting numerics')
    # numerics
    num_cols = ['session_id_size', 'timestamp_dwell_time_prior_clickout', 'last_ref_ind']
    for c in num_cols:
        train[c] = train[c].fillna(-1)

    numerics = train[num_cols].values
    train = train.drop(num_cols, axis=1)

    logger.info('Getting targets')
    # TARGETS
    targets = train['target'].values
    del train['target']
    return numerics, impressions, prices, cfilters, targets


def train(numerics, impressions, prices, cfilters, targets):
    # grab some info on n_cfs
    n_cfs = len(np.load('./cache/filters_mapping.npy').item())
    logger.info(f'Number of unique current_filters is: {n_cfs}')

    batch_size = 128
    n_epochs = 10

    skf = StratifiedKFold(n_splits=6)

    for trn_ind, val_ind in skf.split(targets, targets):
        trn_numerics, val_numerics = numerics[trn_ind], numerics[val_ind]
        trn_imp, val_imp = impressions[trn_ind], impressions[val_ind]
        trn_price, val_price = prices[trn_ind], prices[val_ind]
        trn_cfilter, val_cfilter = cfilters[trn_ind], cfilters[val_ind]
        y_trn, y_val = targets[trn_ind], targets[val_ind]

        # create data generator numerics, impressions, prices, cfilters, targets, batchsize
        # return [numerics_batch, impressions_batch, prices_batch[:, :, None], cfilters_batch]
        train_gen = iterate_minibatches(trn_numerics, trn_imp, trn_price, trn_cfilter, y_trn,
                                        batch_size, shuffle=True)

        val_gen = iterate_minibatches(val_numerics, val_imp, val_price, val_cfilter, y_val,
                                      batch_size, shuffle=False)
        #     TEMP
        #     del impressions, prices, cities, platforms, devices
        #     gc.collect()

        # =====================================================================================
        # create model
        model = build_model(n_cfs, batch_size)

        # print out model info
        nparams = model.count_params()
        logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
                     f'| numer of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
        logger.info(f'{model.summary()}')
        check_dir('./models')
        plot_model(model, to_file='./models/model.png')
        # add some callbacks
        model_file = 'test.model'
        callbacks = [ModelCheckpoint(model_file, save_best_only=True, verbose=1)]
        log_dir = "logs/{}".format(dt.now().strftime('%m-%d-%H-%M'))
        tb = TensorBoard(log_dir=log_dir, write_graph=True, write_grads=True)
        callbacks.append(tb)
        # simple early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1)
        callbacks.append(es)
        # rp
        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100, verbose=1)
        callbacks.append(rp)

        history = model.fit_generator(train_gen,
                                      steps_per_epoch=len(y_trn) // batch_size,
                                      epochs=n_epochs,
                                      verbose=1,
                                      callbacks=callbacks,
                                      validation_data=val_gen,
                                      validation_steps=len(y_val) // batch_size)

        # make prediction
        #      [numerics_batch, impressions_batch, prices_batch[:, :, None], cfilters_batch]
        trn_pred = model.predict(x=[trn_numerics, trn_imp, trn_price[:, :, None], trn_cfilter], batch_size=1024)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = model.predict(x=[val_numerics, val_imp, val_price[:, :, None], val_cfilter], batch_size=1024)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        val_mrr = np.mean(1 / (val_pred_label + 1))
        print(f'train mrr: {trn_mrr:.2f} | val mrr: {val_mrr:.2f}')

        break


if __name__ == '__main__':
    numerics, impressions, prices, cfilters, targets = get_data(nrows=1000000)
    train(numerics, impressions, prices, cfilters, targets)