import numpy as np
from datetime import datetime as dt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model

from utils import get_logger, check_dir
from nn_model_simple import build_model
from create_nn_train_input import create_train_inputs
from create_nn_test_input import create_test_inputs


logger = get_logger('train_nn')

TO_DO = ('1) maybe fillna -1 is too overwhelming for last_reference_id_index if normalized \n'
         '2) session_id size and dwell_time prior last click maybe need normalization in scale or maybe add batchnorm')

logger.info(TO_DO)


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


def train(numerics, impressions, prices, cfilters, targets):
    # grab some info on n_cfs
    n_cfs = len(np.load('./cache/filters_mapping.npy').item())
    logger.info(f'Number of unique current_filters is: {n_cfs}')

    batch_size = 128
    n_epochs = 5

    skf = StratifiedKFold(n_splits=6)
    models = []
    for fold, (trn_ind, val_ind) in enumerate(skf.split(targets, targets)):
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
        model_file = f'./models/cv{fold}.model'
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
        models.append(model)
        break
    return models


if __name__ == '__main__':
    # first create training inputs
    numerics, impressions, prices, cfilters, targets = create_train_inputs(nrows=500000, recompute=False)
    # train the model
    models = train(numerics, impressions, prices, cfilters, targets)
    # get the test inputs
    numerics, impressions, prices, cfilters = create_test_inputs(recompute=False)
    # make predictions on test
    check_dir('./subs')
    logger.info('Load test sub csv')
    test_sub = pd.read_csv('./cache/test_sub.csv')
    for m, model in enumerate(models):
        test_sub_m = test_sub.copy()

        logger.info(f'Generating submission from model {m}')
        test_pred = model.predict(x=[numerics, impressions, prices[:, :, None], cfilters], batch_size=2048)
        test_pred_label = np.argsort(test_pred)[:, ::-1]
        test_impressions = np.array(list(test_sub_m['impressions']))
        test_impressions_pred = test_impressions[np.arange(len(test_impressions))[:, None], test_pred_label]
        test_sub_m['recommendations'] = test_impressions_pred
        test_sub_m.to_csv('sub_m.csv', index=False)
