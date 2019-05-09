import os
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import plot_model
from keras.models import load_model

from utils import get_logger, check_dir
from nn_model import build_model
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


def plot_hist(pred_label, true_label, name):
    _ = plt.hist(pred_label, bins=50, label=f'{name}_pred', alpha=0.7)
    _ = plt.hist(true_label, bins=50, label = f'{name} label', alpha=0.7)
    _ = plt.legend()
    check_dir('./plots')
    plt.savefig(f'./plots/{name}_hist.png')
    plt.gcf().clear()


def confusion_matrix(y_pred, y_true, name, normalize='row', level=0, log_scale=False):
    compare = pd.DataFrame({'prediction': y_pred, 'y_true': y_true})
    counts = compare.groupby('y_true')['prediction'].value_counts()
    mat = counts.unstack(level=level)
    mat.fillna(0, inplace=True)

    if normalize == 'row':
        row_sum = mat.sum(axis=1)
        mat = mat.div(row_sum, axis=0)
        log_scale = False
    elif normalize == 'column':
        col_sum = mat.sum(axis=0)
        mat = mat.div(col_sum, axis=1)
        log_scale = False
    # plot
    fig = plt.figure(figsize=(35, 10))
    ax = fig.add_subplot(111)
    if log_scale:
        cax = ax.matshow(np.log1p(mat), interpolation='nearest')  # , cmap='coolwarm')#, aspect='auto')
    else:
        cax = ax.matshow(mat, interpolation='nearest')  # , cmap='coolwarm')#, aspect='auto')
    fig.colorbar(cax)
    ax.set_xlabel(f'{mat.columns.name}')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(f'{mat.index.name}')

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns.astype(str)), rotation=90)
    ax.set_yticks(np.arange(mat.shape[0]))
    _ = ax.set_yticklabels(list(mat.index.astype(str)))
    check_dir('./plots')
    plt.savefig(f'./plots/{name}_confusion_matrix.png')
    plt.gcf().clear()


def train(numerics, impressions, prices, cfilters, targets, retrain=False):
    # grab some info on n_cfs
    n_cfs = len(np.load('./cache/filters_mapping.npy').item())
    logger.info(f'Number of unique current_filters is: {n_cfs}')

    batch_size = 128
    n_epochs = 100

    skf = StratifiedKFold(n_splits=6)
    models = []
    for fold, (trn_ind, val_ind) in enumerate(skf.split(targets, targets)):
        trn_numerics, val_numerics = numerics[trn_ind], numerics[val_ind]
        trn_imp, val_imp = impressions[trn_ind], impressions[val_ind]
        trn_price, val_price = prices[trn_ind], prices[val_ind]
        trn_cfilter, val_cfilter = cfilters[trn_ind], cfilters[val_ind]
        y_trn, y_val = targets[trn_ind], targets[val_ind]

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
            model = build_model(n_cfs, batch_size)

            # print out model info
            nparams = model.count_params()
            logger.info((f'train len: {len(y_trn):,} | val len: {len(y_val):,} '
                         f'| numer of parameters: {nparams:,} | train_len/nparams={len(y_trn) / nparams:.5f}'))
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
    return models


if __name__ == '__main__':
    # setup = {'nrows': 1000000}
    setup = {'nrows': None}

    # first create training inputs
    numerics, impressions, prices, cfilters, targets = create_train_inputs(nrows=setup['nrows'], recompute=False)
    # train the model
    models = train(numerics, impressions, prices, cfilters, targets, retrain=False)
    # get the test inputs
    numerics, impressions, prices, cfilters = create_test_inputs(recompute=False)
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

