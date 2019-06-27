import os
import re
import time
import gc
import pprint
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ast import literal_eval

from sklearn.model_selection import KFold, ShuffleSplit
import catboost as cat

from create_model_inputs import create_model_inputs, CATEGORICAL_COLUMNS
from utils import get_logger, get_data_path, check_gpu
from plots import plot_hist, confusion_matrix, plot_imp_cat, compute_shap_multi_class


logger = get_logger('train_cat')
Filepath = get_data_path()
# random splitting for cross-validation
RS = 42

# catboost GPU does not support custom metric


def compute_mrr(pred, y_true):
    pred_label = np.where(np.argsort(pred)[:, ::-1] == y_true.reshape(-1, 1))[1]
    mrr = np.mean(1 / (pred_label + 1))
    return mrr


def log_median(df, col):
    df[col] = np.log((1+df[col])/(1+np.median(df[col])))


def cat_preprocess(df):
    # specify some columns that we do not want in training
    cf_cols = [i for i in df.columns if 'current_filters' in i]
    price_cols = [i for i in df.columns if re.match(r'prices_\d', i)]

    drop_cols = cf_cols + price_cols + ['country', 'platform']
    drop_cols = [col for col in df.columns if col in drop_cols]
    # drop col
    logger.info(f'Preliminary Drop columns:\n {drop_cols}')
    df.drop(drop_cols, axis=1, inplace=True)

    # fillna
    prev_cols = [i for i in df.columns if 'prev' in i]
    df.loc[:, prev_cols] = df.loc[:, prev_cols].fillna(0)

    # some transformation
    to_log_median_cols = ['last_duration', 'session_duration', 'session_size', 'n_imps',
                          'mean_price', 'median_price', 'std_price', 'n_cfs',
                          'step', 'step_no_gap']
    prev_cols = [i for i in df.columns if 'prev' in i]
    to_log_median_cols.extend(prev_cols)

    logger.info(f'Performing log_median columns on:\n{np.array(to_log_median_cols)}')
    logger.warning('THIS PROBABLY WILL MAKE VALIDATION PERFORMANCE LOOK BETTER THAT IT ACTUALLY IS!!!')
    for col in tqdm(to_log_median_cols):
        log_median(df, col)

    # price_bins_cols = [i for i in df.columns if 'bin' in i]
    # df[price_bins_cols] = df[price_bins_cols]/5

    logger.info(f'COLUMNS:\n{list(df.columns)}')

    filename = 'train_inputs_cat.snappy'
    df.to_parquet(os.path.join(Filepath.cache_path, filename), index=False)
    return df


def train(train_df, only_last=False):
    # path to where model is saved
    model_path = Filepath.model_path

    # temp
    train_clean = pd.read_parquet('./cache/basic_preprocessed_train_5000000.snappy')
    length = train_clean.groupby('session_id')['session_id'].size()
    train_df['length'] = train_df['session_id'].map(length)
    del train_clean
    gc.collect()
    assert train_df['length'].isna().sum() == 0

    # if only use the last row of train_df to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_df = train_df.groupby('session_id').last().reset_index(drop=False)

    # a bit processing
    cat_preprocess(train_df)

    train_cols = train_df.columns
    logger.info(f'Columns used for training: {train_cols.values}')
    used_categorical_cols = train_cols[train_cols.isin(CATEGORICAL_COLUMNS)]
    logger.info(f'Categorical columns in training: {used_categorical_cols.values}')

    # grab unique session ids and use this to split, so that train_df with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_df['session_id'].unique()

    # kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    ss = ShuffleSplit(n_splits=5, test_size=0.15, random_state=RS)

    # record classifiers and mrr each training
    cv_clfs = []
    cv_mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(ss.split(unique_session_ids)):
        logger.info(f'Training fold {fold}: train ids len={len(trn_ind):,} | val ids len={len(val_ind):,}')
        # get session_id used for train
        trn_ids = unique_session_ids[trn_ind]
        trn_mask = train_df['session_id'].isin(trn_ids)
        logger.info(f'Training fold {fold}: train len={trn_mask.sum():,} | val len={(~trn_mask).sum():,}')

        x_trn, x_val = (train_df[trn_mask].reset_index(drop=True),
                        train_df[~trn_mask].reset_index(drop=True))

        # split the training into two models, first with only session_size=1 second with > 1
        trn_ones_mask = x_trn['length'] == 1
        val_ones_mask = x_val['length'] == 1
        n_ones_trn, n_ones_val = trn_ones_mask.sum(), val_ones_mask.sum()
        n_more_trn, n_more_val = (~trn_ones_mask).sum(), (~val_ones_mask).sum()
        logger.info(f'Train has {n_ones_trn:,} ones session and {n_more_trn:,} more | '
                    f'Val has {n_ones_val:,} ones session and {n_more_val:,} more')

        # train
        x_trn_ones, x_trn_more = (x_trn[trn_ones_mask].reset_index(drop=True),
                                  x_trn[~trn_ones_mask].reset_index(drop=True))
        # val
        x_val_ones, x_val_more = (x_val[val_ones_mask].reset_index(drop=True),
                                  x_val[~val_ones_mask].reset_index(drop=True))

        # for validation only last row is needed
        x_val_more = x_val_more.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn_ones, y_val_ones = x_trn_ones['target'].values, x_val_ones['target'].values
        y_trn_more, y_val_more = x_trn_more['target'].values, x_val_more['target'].values

        remove_ones_cols = [col for col in x_trn.columns if 'prev' in col]
        remove_ones_cols += ['last_action_type', 'last_reference_relative_loc', 'last_duration', 'imp_changed',
                             'fs', 'sort_order']
        remove_ones_cols += ['session_id', 'length', 'target']
        x_trn_ones.drop(remove_ones_cols, axis=1, inplace=True)
        x_val_ones.drop(remove_ones_cols, axis=1, inplace=True)

        x_trn_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)
        x_val_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)

        # get categorical index
        cat_ind_ones = [k for k, v in enumerate(x_trn_ones.columns) if v in CATEGORICAL_COLUMNS]
        cat_ind_more = [k for k, v in enumerate(x_trn_more.columns) if v in CATEGORICAL_COLUMNS]

        # =====================================================================================
        # train model
        device = 'GPU' if check_gpu() else 'CPU'
        params_ones = {'loss_function': 'MultiClass',
                       'custom_metric': ['Accuracy'],
                       'eval_metric': 'MultiClass',
                       'iterations': 3000,
                       'learning_rate': 0.02,
                       'early_stopping_rounds': 100,
                       # SymmetricTree, Depthwise, Lossguide (lossguide seems like not implemented)
                       'grow_policy': 'SymmetricTree',
                       'depth': 5,
                       'bootstrap_type': 'Bayesian',  # Poisson, Bayesian, Bernoulli
                       # 'subsample': 0.8,
                       # 'bagging_temperature': 1,
                       'l2_leaf_reg': 1,
                       'random_strength': 1,
                       # 'border_count': 100,
                       'task_type': device}

        # train first he ones
        clf_one = cat.CatBoostClassifier(**params_ones)
        clf_one.fit(x_trn_ones, y_trn_ones,
                    cat_features=cat_ind_ones,
                    eval_set=(x_val_ones, y_val_ones),
                    verbose=100,
                    plot=False)

        params_more = {'loss_function': 'MultiClass',
                       'custom_metric': ['Accuracy'],
                       'eval_metric': 'MultiClass',
                       'iterations': 10000,
                       'learning_rate': 0.02,
                       'early_stopping_rounds': 100,
                       # SymmetricTree, Depthwise, Lossguide (lossguide seems like not implemented)
                       'grow_policy': 'SymmetricTree',
                       'depth': 8,
                       'bootstrap_type': 'Bayesian',  # Poisson, Bayesian, Bernoulli
                       # 'subsample': 0.8,
                       # 'bagging_temperature': 1,
                       'l2_leaf_reg': 1,
                       'random_strength': 1,
                       # 'border_count': 100,
                       'task_type': device}

        clf_more = cat.CatBoostClassifier(**params_more)
        clf_more.fit(x_trn_more, y_trn_more,
                     cat_features=cat_ind_more,
                     eval_set=(x_val_more, y_val_more),
                     verbose=100,
                     plot=False)

        # make prediction
        x_trn = train_df[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        # train
        trn_ones_mask = x_trn['length'] == 1
        x_trn_ones, x_trn_more = (x_trn[trn_ones_mask].reset_index(drop=True),
                                  x_trn[~trn_ones_mask].reset_index(drop=True))

        y_trn_ones, y_trn_more = x_trn_ones['target'].values, x_trn_more['target'].values
        x_trn_ones.drop(remove_ones_cols, axis=1, inplace=True)
        x_trn_more.drop(['session_id', 'length', 'target'], axis=1, inplace=True)

        trn_pred_ones, trn_pred_more = clf_one.predict_proba(x_trn_ones), clf_more.predict_proba(x_trn_more)
        trn_pred = np.concatenate((trn_pred_ones, trn_pred_more), axis=0)
        y_trn = np.concatenate((y_trn_ones, y_trn_more), axis=0)
        trn_mrr = compute_mrr(trn_pred, y_trn)
        trn_mrr_ones = compute_mrr(trn_pred_ones, y_trn_ones)
        trn_mrr_more = compute_mrr(trn_pred_more, y_trn_more)

        val_pred_ones, val_pred_more = clf_one.predict_proba(x_val_ones), clf_more.predict_proba(x_val_more)
        val_pred = np.concatenate((val_pred_ones, val_pred_more), axis=0)
        y_val = np.concatenate((y_val_ones, y_val_more), axis=0)
        val_mrr = compute_mrr(val_pred, y_val)
        val_mrr_ones = compute_mrr(val_pred_ones, y_val_ones)
        val_mrr_more = compute_mrr(val_pred_more, y_val_more)
        logger.info(f'\ntrain mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}\n'
                    f'train mrr ones: {trn_mrr_ones:.4f} | val mrr ones: {val_mrr_ones:.4f}\n'
                    f'train mrr more: {trn_mrr_more:.4f} | val mrr more: {val_mrr_more:.4f}\n')

        np.save(os.path.join(model_path, f'lgb_val_{fold}_pred.npy'), val_pred)

        cv_clfs.append(clf)
        cv_mrrs.append((trn_mrr, val_mrr))

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return cv_clfs, cv_mrrs


if __name__ == '__main__':
    setup = {'nrows': 5000000,
             'recompute_train': False,
             'add_test': False,
             'only_last': False,
             'retrain': True,
             'recompute_test': False}

    device = 'GPU' if check_gpu() else 'CPU'
    params = {'loss_function': 'MultiClass',
              'custom_metric': ['Accuracy'],
              'eval_metric': 'MultiClass',
              'iterations': 500,
              'learning_rate': 0.03,
              'early_stopping_rounds': 100,
              # SymmetricTree, Depthwise, Lossguide (lossguide seems like not implemented)
              'grow_policy': 'SymmetricTree',
              'depth': 8,
              'bootstrap_type': 'Bayesian',  # Poisson, Bayesian, Bernoulli
              # 'subsample': 0.8,
              # 'bagging_temperature': 1,
              # 'l2_leaf_reg': 10,
              # 'random_strength': 10,
              # 'border_count': 100,
              'task_type': device}

    if params['grow_policy'] == 'Lossguide':
        # Can be used only with the Lossguide growing policy.
        params['max_leaves'] = 31

    if device == 'GPU':
        # Can be used only with the Lossguide and Depthwise growing policies.
        if params['grow_policy'] == 'SymmetricTree':
            logger.warning('min_data_in_leaf is not used in SymmetricTree')
        else:
            params['min_data_in_leaf'] = 5
    else:
        params['rsm'] = 0.9

    logger.info(f"\nSetup\n{'='*20}\n{pprint.pformat(setup)}\n{'='*20}")
    logger.info(f"\nParams\n{'='*20}\n{pprint.pformat(params)}\n{'='*20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], padding_value=np.nan,
                                       add_test=setup['add_test'], recompute=setup['recompute_train'])
    # train the model
    models, mrrs = train(train_inputs, params=params, only_last=setup['only_last'], retrain=setup['retrain'])
    mean_train_mrr = np.mean([mrr[0] for mrr in mrrs])
    mean_val_mrr = np.mean([mrr[1] for mrr in mrrs])
    # get the test inputs
    test_inputs = create_model_inputs(mode='test', padding_value=np.nan, recompute=setup['recompute_test'])

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
        logger.info(f'Generating predictions from model {c}')
        test_pred = clf.predict_proba(test_inputs)
        # save prediction
        np.save(os.path.join(Filepath.sub_path, 'cat_test_0_pred.npy'), test_pred)
        test_predictions.append(test_pred)

    logger.info('Generating submission by averaging cv predictions')
    test_predictions = np.array(test_predictions).mean(axis=0)
    test_pred_label = np.argsort(test_predictions)[:, ::-1]
    np.save(os.path.join(Filepath.sub_path, f'test_pred_label.npy'), test_pred_label)

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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'cat_sub_{current_time}_{mean_train_mrr:.4f}_{mean_val_mrr:.4f}.csv'),
                    index=False)
    logger.info('Done all')

