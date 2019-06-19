import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from tqdm import tqdm

from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras import optimizers
from keras.utils import plot_model
from keras.models import load_model
from model import build_ensemble

from create_model_inputs import create_model_inputs, CATEGORICAL_COLUMNS
from utils import get_logger, get_data_path, check_gpu, ignore_warnings

ignore_warnings()

logger = get_logger('train_lgb')
Filepath = get_data_path()
RS = 42

setup = {'nrows': 5000000,
         'recompute_train': False,
         'add_test': False,
         'only_last': False,
         'retrain': True,
         'recompute_test': False}
train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], padding_value=np.nan,
                                   add_test=setup['add_test'], recompute=setup['recompute_train'])

n_fold = 1 # 5
test_fraction = 0.15
unique_session_ids = train_inputs['session_id'].unique()
ss = ShuffleSplit(n_splits=n_fold, test_size=test_fraction, random_state=RS)

# get path where predictions were saved
prediction_path = Filepath.model_path


def get_pred(model, mode, fold):
    return np.load(os.path.join(prediction_path, f'{model}_{mode}_{fold}_pred.npy'))


for fold, (trn_ind, val_ind) in enumerate(ss.split(unique_session_ids)):
    logger.info(f'Training fold {fold}: train ids len={len(trn_ind):,} | val ids len={len(val_ind):,}')
    # get session_id used for train
    trn_ids = unique_session_ids[trn_ind]
    trn_mask = train_inputs['session_id'].isin(trn_ids)
    logger.info(f'Training fold {fold}: train len={trn_mask.sum():,} | val len={(~trn_mask).sum():,}')

    y_trn = train_inputs[trn_mask]
    y_trn = y_trn.groupby('session_id').last().reset_index(drop=False)
    y_trn = y_trn['target'].values

    y_val = train_inputs[~trn_mask]
    y_val = y_val.groupby('session_id').last().reset_index(drop=False)
    y_val = y_val['target'].values

    # load train
    trn_lgb = get_pred('lgb', 'trn', fold)
    trn_cat = get_pred('cat', 'trn', fold)
    trn_nn = get_pred('nn', 'trn', fold)
    # load val
    val_lgb = get_pred('lgb', 'val', fold)
    val_cat = get_pred('cat', 'val', fold)
    val_nn = get_pred('nn', 'val', fold)
    print('trn_lgb:', trn_lgb.shape, 'cat', trn_cat.shape, 'nn', trn_nn.shape)
    x_trn = np.concatenate([trn_lgb, trn_cat, trn_nn], axis=1)
    x_val = np.concatenate([val_lgb, val_cat, val_nn], axis=1)
    print('x_trn:', x_trn.shape)
    print(y_trn.shape)

    # train
    # # clf = GaussianNB()
    # # clf = LinearSVC(penalty='l2', random_state=0, tol=1e-5, probability=True)
    # clf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=0, n_jobs=-1)
    # # clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    # clf.fit(x_trn, y_trn)
    # trn_pred = clf.predict_proba(x_trn)
    # val_pred = clf.predict_proba(x_val)
    # print(trn_pred.shape)

    model = build_ensemble()
    nparams = model.count_params()
    # opt = optimizers.Adam(lr=params['learning_rate'])
    opt = optimizers.Adagrad(lr=0.01)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    # model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    # model.compile(optimizer=opt, loss=custom_objective, metrics=['categorical_crossentropy'])

    logger.info(f'{model.summary()}')

    history = model.fit(x=x_trn,
                        y=y_trn,
                        batch_size=1024,
                        epochs=50,
                        verbose=1,
                        validation_data=(x_val, y_val))

    trn_pred = model.predict(x=x_trn, batch_size=1024)
    val_pred = model.predict(x=x_val, batch_size=1024)

    # pred label
    # trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
    # val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
    trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
    val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]

    # calculate mrr
    trn_mrr = np.mean(1 / (trn_pred_label + 1))
    val_mrr = np.mean(1 / (val_pred_label + 1))

    logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')






    # y_trn_binary = to_categorical(y_trn)
    # y_val_binary = to_categorical(y_val)
    # trn_pred, val_pred = [], []
    # for i in tqdm(range(25)):
    #     print(list(range(i, 75, 25)))
    #     x_trn_ = x_trn[:, list(range(i, 75, 25))]
    #     x_val_ = x_val[:, list(range(i, 75, 25))]
    #
    #     y_trn_ = y_trn_binary[:, i]
    #     clf = LogisticRegression().fit(x_trn_, y_trn_)
    #     trn_pred_ = clf.predict_proba(x_trn_)
    #     val_pred_ = clf.predict_proba(x_val_)
    #
    #     trn_pred.append(trn_pred_)
    #     val_pred.append(val_pred_)
    # trn_pred = np.concatenate(trn_pred, axis=1)
    # val_pred = np.concatenate(val_pred, axis=1)
    # print(trn_pred.shape)
    # print(trn_pred[0, :])
    # # pred label
    # trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
    # val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
    #
    # # calculate mrr
    # trn_mrr = np.mean(1 / (trn_pred_label + 1))
    # val_mrr = np.mean(1 / (val_pred_label + 1))
    #
    # logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')



