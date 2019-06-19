import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import RidgeClassifier, LogisticRegression

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
    y_val = train_inputs[~trn_mask]['target'].values

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
    # clf = RidgeClassifier().fit(x_trn, y_trn)
    clf = LogisticRegression(multi_class='multinomial', solver='newton-cg')
    clf.fit(x_trn, y_trn)
    trn_pred = clf.predict_proba(x_trn)
    val_pred = clf.predict_proba(x_val)
    print(trn_pred.shape)
    # pred label
    trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
    val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]

    # calculate mrr
    trn_mrr = np.mean(1 / (trn_pred_label + 1))
    val_mrr = np.mean(1 / (val_pred_label + 1))

    logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')


