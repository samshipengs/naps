# import multiprocessing as mp
import time
# import os
# import gc
# import datetime
import pandas as pd
import numpy as np
# from tqdm import tqdm
from sklearn.metrics import log_loss, auc, roc_curve, f1_score, average_precision_score, mean_squared_error
# from sklearn.model_selection import StratifiedKFold
import catboost as cat
from utils import check_gpu, check_dir, plot_imp
from reduce_memory import reduce_numeric_mem_usage, reduce_object_mem_usage


def categorize(xtrain, xval, cat_fts, xtest=None):
    # cat_fts = ['city_get_first', 'platform_get_first', 'device_get_first', 'item_id', 'location']
    # convert to categorical
    for c in cat_fts:
        print(f'>>> {c}')
        if xtest is not None:
            maps = list(set(list(xtrain[c].unique()) + list(xval[c].unique()) + list(xtest[c].unique())))
        else:
            maps = list(set(list(xtrain[c].unique()) + list(xval[c].unique())))
            #     maps = xtrain[c].unique()

        mapper = dict(zip(maps, np.arange(len(maps), dtype=int)))
        if c == 'item_id':
            print('ITEM_ID reverse mapper is getting saved for output test purpose')
            mapper_reverse = {v: k for k, v in mapper.items()}
            np.save('./data/item_id_mapper_reverse.npy', mapper_reverse)
        xtrain[c] = xtrain[c].map(mapper)
        xval[c] = xval[c].map(mapper)
        if xtest is not None:
            xtest[c] = xtest[c].map(mapper)
    print('done categorizing')


def train_model(xtrain, xval, cat_fts, params):
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
    print('Grab feature importance for both train and val')
    # get feature importance
    trn_imp = clf.get_feature_importance(data=cat.Pool(data=xtrain, cat_features=categorical_ind),
                                         prettified=True)
    val_imp = clf.get_feature_importance(data=cat.Pool(data=xval, cat_features=categorical_ind),
                                         prettified=True)
    plot_imp(trn_imp, 'train')
    plot_imp(val_imp, 'val')
    print('Done feature imp')

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
    xval['target'] = y_val
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
    t_int = time.time()
    fprint = lambda msg: print(f'{msg} {"="*20} time elapsed = {(time.time()-t_int)/60:.2f} mins')
    fprint('Load train data')
    train = pd.read_hdf('./data/train.h5', 'train')

    train_val_split = 0.1
    # split out validation from the latest
    sort_ts = train.sort_values(by='ts')['ts']
    trn_sids = sort_ts.iloc[:int(len(train)*train_val_split)].index
    val_sids = sort_ts.iloc[-int(len(train)*train_val_split):].index

    val_sids = list(set(val_sids.unique()) - set(trn_sids.unique()))
    val_mask = train.index.isin(val_sids)
    xtrain = train[~val_mask]
    xval = train[val_mask]
    fprint(f'xtrain shape: {xtrain.shape}')
    fprint(f'xval shape: {xval.shape}')

    # fprint('Load test')
    # xtest = pd.read_hdf('./data/test.h5', 'xtest')
    # fprint('categorizing features')
    cat_fts = ['city_get_last', 'platform_get_last', 'device_get_last', 'item_id', 'location']
    #
    # categorize(xtrain, xval, cat_fts, xtest)

    # fprint('reducing memory')
    # reduce_numeric_mem_usage(xtrain)
    # reduce_numeric_mem_usage(xval)
    # reduce_numeric_mem_usage(xtest)

    # xtrain.set_index('session_id', inplace=True)
    # xval.set_index('session_id', inplace=True)
    # xtest.set_index('session_id', inplace=True)

    fprint('Start training')
    device = 'GPU' if check_gpu() else 'CPU'
    params = {'iterations': 300,
              'learning_rate': 0.02,
              'depth': 8,
              'task_type': device}
    clf, categorical_ind, mrr = train_model(xtrain, xval, cat_fts, params)

    # fprint('Make prediction on test set')
    # # pred xtest
    # test_pred = clf.predict_proba(xtest.values)[:, 1]
    # xtest['pred'] = test_pred
    # item_mapper = np.load('./data/item_id_mapper_reverse.npy').item()
    # xtest['item_id'] = xtest['item_id'].map(item_mapper)
    # test_imps_pred = xtest.groupby(level=0).apply(output_impressions)
    # test_imps_pred = test_imps_pred.reset_index(name='recommendation')
    # test_imps_pred.to_csv('./data/test_imps_pred.csv', index=False)
    # # read sub
    # sub = pd.read_csv('./data/submission_popular.csv')
    # sub = pd.merge(sub, test_imps_pred, how='left', on='session_id')
    # sub.to_csv('./data/sub.csv', index=False)
    #
    # sub.drop('item_recommendations', axis=1, inplace=True)
    # sub.rename(columns={'recommendation': 'item_recommendations'}, inplace=True)
    # sub.to_csv(f'./data/sub_mrr_{mrr:.4f}.csv', index=False)

    fprint('DONE')


if __name__ == '__main__':
    run_pipeline()
