import multiprocessing as mp
import time
import os
import gc
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss, auc, roc_curve, f1_score, average_precision_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
import catboost as cat
from utils import check_gpu
from reduce_memory import reduce_numeric_mem_usage, reduce_object_mem_usage


def categorize(xtrain, xval, cat_fts, xtest=None):
    # cat_fts = ['city_get_first', 'platform_get_first', 'device_get_first', 'item_id', 'location']
    # convert to categorical
    for c in cat_fts:
        print(f'>>> {c}')
        maps = list(set(list(xtrain[c].unique()) + list(xval[c].unique())))
        #     maps = xtrain[c].unique()
        mapper = dict(zip(maps, np.arange(len(maps), dtype=int)))
        xtrain[c] = xtrain[c].map(mapper)
        xval[c] = xval[c].map(mapper)
        if xtest is not None:
            xtest[c] = xval[c].map(mapper)
    print('done categorizing')


def train_model(xtrain, xval, cat_fts, params):
    # params = {'iterations': 1000,
    #           'learning_rate': 0.02,
    #           'depth': 8,
    #           'task_type': 'CPU'}
    # #          'task_type': 'GPU'}

    y_trn = xtrain['target'].values
    y_val = xval['target'].values
    del xtrain['target'], xval['target']

    categorical_ind = [k for k, v in enumerate(xtrain.columns) if v in cat_fts]

    # train model
    clf = cat.CatBoostClassifier(**params)
    clf.fit(xtrain.values, y_trn,
            cat_features=categorical_ind,
            eval_set=(xval.values, y_val),
            early_stopping_rounds=100,
            verbose=100,
            plot=False)
    print('Done!')
    # try to save model
    # model_path = './cat_model'
    # cat.save_model(clf, model_path)

    # make prediction on validation set
    val_pred = clf.predict_proba(xval.values)[:, 1]
    logloss_i = log_loss(y_val, val_pred)
    # compute roc auc
    fpr, tpr, thresholds = roc_curve(y_val, val_pred, pos_label=1)
    auc_i = auc(fpr, tpr)
    # compute map
    map_i = average_precision_score(y_val, val_pred)
    print('logloss={0:.4f} | map={1:.4f} | auc={2:.4f}'.format(logloss_i, map_i, auc_i))

    # mrr
    print('reciproical rank for validation set')
    xval['pred'] = val_pred
    xval['target'] = y_val.values
    val_rr = xval.groupby(level=0).apply(reciprocal_rank)
    mrr = (1/val_rr[val_rr != 0]).mean()
    print(f'Mean reciporical rank on validation set: {mrr:.4f}')
    return clf, categorical_ind, mrr


def reciprocal_rank(df):
    pred_list = df['pred'].values
    sorted_ind = np.argsort(pred_list)[::-1]
    ranked_items = list(df['item_id'].iloc[sorted_ind].values)
    try:
        target = df.loc[df['target'] == 1, 'item_id'].values[0]
        rank = ranked_items.index(target)
    except:
        return 0
    return rank+1


def output_impressions(df):
    pred_list = df['pred'].values
    sorted_ind = np.argsort(pred_list)[::-1]
    ranked_items = list(df['item_id'].iloc[sorted_ind].values)
    ranked_items_str = [str(i) for i in ranked_items]
    return ' '.join(ranked_items_str)


def run_pipeline():
    fprint = lambda msg: print(msg + '='*20)
    fprint('Load train data')
    xtrain = pd.read_hdf('./data/train.h5', 'xtrain')
    xval = pd.read_hdf('./data/train.h5', 'xval')
    fprint('Load test')
    xtest = pd.read_hdf('./data/test.h5', 'xtest')
    fprint('categorizing features')
    cat_fts = ['city_get_first', 'platform_get_first', 'device_get_first', 'item_id', 'location']
    categorize(xtrain, xval, cat_fts, xtest)

    fprint('reducing memory')
    reduce_numeric_mem_usage(xtrain)
    reduce_numeric_mem_usage(xval)
    reduce_numeric_mem_usage(xtest)

    xtrain.set_index('session_id', inplace=True)
    xval.set_index('session_id', inplace=True)
    xtest.set_index('session_id', inplace=True)

    fprint('Start training')
    device = 'GPU' if check_gpu() else 'CPU'
    params = {'iterations': 1000,
              'learning_rate': 0.02,
              'depth': 8,
              'task_type': device}
    clf, categorical_ind, mrr = train_model(xtrain, xval, cat_fts, params)

    fprint('Make prediction on test set')
    # pred xtest
    test_pred = clf.predict_proba(xtest)[:, 1]
    xtest['pred'] = test_pred
    test_imps_pred = xtest.groupby(level=0).apply(output_impressions)
    test_imps_pred = test_imps_pred.reset_index(name='recommendation')
    # read sub
    sub = pd.read_csv('./data/submission_popular.csv')
    sub = pd.merge(sub, test_imps_pred, how='left', left_on='item_recommendations', right_on='recommendation')
    sub.to_csv('./sub.csv')
    fprint('DONE')


if __name__ == '__main__':
    run_pipeline()