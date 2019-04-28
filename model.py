"""
Train model and perform necessary steps such as convert categorical column to natural number and provide categorical
index for lightgbm and catboost
"""
import os
import gc
import numpy as np
import pandas as pd

from utils import load_data, check_dir, check_gpu, get_logger, plot_imp_cat, plot_imp_lgb
from reduce_memory import reduce_numeric_mem_usage
from data_pipeline import combine_inputs
from sklearn.model_selection import StratifiedKFold, KFold
import catboost as cat
import lightgbm as lgb


logger = get_logger('model')


# encode city, platform and device
def categorize(df, cols):
    for col in cols:
        logger.info(f'categorizing {col}')
        unique_values = df[col].unique()
        mapping = {v: k for k, v in enumerate(unique_values)}
        df[col] = df[col].map(mapping)


def train_cat(params, df=None, split_mode='target', n_splits=5):
    if df is None:
        filename = os.path.join('./cache/combined_inputs.parquet')
        logger.info(f'No training dataframe is provided, trying to load {filename} from disk')
        df = pd.read_parquet(filename)
    logger.info(df['target'].value_counts())

    # categorize
    cat_fts = ['city', 'platform', 'device', 'action_type', 'impression']
    logger.info(f'Converting following to natural number: {cat_fts}')
    categorize(df, cat_fts)

    if split_mode == 'target':
        logger.info('Splitting by stratified targets')
        folds = StratifiedKFold(n_splits=n_splits)
        splits = df.target.values
    else:
        logger.info('Splitting by session_ids')
        folds = KFold(n_splits=n_splits)
        sids = df['session_id'].unique()
        splits = sids
        df_sids = df['session_id']

    # create oof
    oof = df[['session_id', 'impression', 'target']].copy()
    # base as used for each cv metric prints
    base = df[['session_id', 'impression', 'target']].copy()
    del df['session_id']
    # grab the targets
    target = df.target.values
    del df['target']

    # get the column index of the categorical columns
    categorical_ind = [k for k, v in enumerate(df.columns) if v in cat_fts]

    logger.info(f'Start training with cv and params:\n{params}')
    for fold, (trn_ind, val_ind) in enumerate(folds.split(splits, splits)):
        if split_mode == 'target':
            x_trn, y_trn = df.iloc[trn_ind], target[trn_ind]
            x_val, y_val = df.iloc[val_ind], target[val_ind]
            base_trn, base_val = base.iloc[trn_ind].copy(), base.iloc[val_ind].copy()
        else:
            trn_mask = df_sids.isin(sids[trn_ind])
            x_trn, y_trn = df[trn_mask], target[trn_mask]
            x_val, y_val = df[~trn_mask], target[~trn_mask]
            base_trn, base_val = base.loc[trn_mask].copy(), base.loc[~trn_mask].copy()
            # since the split index was based on only ids, we need to give it the actual index for oof
            val_ind = np.where(~trn_mask)

        # train model
        clf = cat.CatBoostClassifier(**params)
        clf.fit(x_trn.values, y_trn,
                cat_features=categorical_ind,
                eval_set=(x_val.values, y_val),
                early_stopping_rounds=100,
                verbose=100,
                plot=False)

        logger.info('Done training, generating predictions for oof, train and val for evaluation purpose')
        # assign prediction
        trn_rped =clf.predict_proba(x_trn.values)[:, 1]
        val_pred = clf.predict_proba(x_val.values)[:, 1]

        oof.loc[oof.index[val_ind], 'prediction'] = val_pred
        base_trn['prediction'] = trn_rped
        base_val['prediction'] = val_pred

        # compute mrr
        logger.info('Computing train mrr')
        trn_mrr = mean_reciprocal_rank(base_trn)
        logger.info('Computing val mrr')
        val_mrr = mean_reciprocal_rank(base_val)

        logger.info('Getting feature importance')
        # get feature importance
        trn_imp = clf.get_feature_importance(data=cat.Pool(data=x_trn, cat_features=categorical_ind),
                                             prettified=True)
        plot_imp_cat(trn_imp, fold, mrr={'train': trn_mrr, 'val': val_mrr})
        logger.info('Done feature imp')
        break

    logger.info('Computing oof mrr')
    mean_reciprocal_rank(oof)
    logger.info('Done all')


def train_lgb(params, df=None, split_mode='target', n_splits=5):
    if df is None:
        filename = os.path.join('./cache/combined_inputs.parquet')
        logger.info(f'No training dataframe is provided, trying to load {filename} from disk')
        df = pd.read_parquet(filename)
    logger.info(df['target'].value_counts())

    # categorize
    cat_fts = ['city', 'platform', 'device', 'action_type', 'impression']
    logger.info(f'Converting following to natural number: {cat_fts}')
    categorize(df, cat_fts)

    if split_mode == 'target':
        logger.info('Splitting by stratified targets')
        folds = StratifiedKFold(n_splits=n_splits)
        splits = df.target.values
    else:
        logger.info('Splitting by session_ids')
        folds = KFold(n_splits=n_splits)
        sids = df['session_id'].unique()
        splits = sids
        df_sids = df['session_id']

    # create oof
    oof = df[['session_id', 'impression', 'target']].copy()
    # base as used for each cv metric prints
    base = df[['session_id', 'impression', 'target']].copy()
    del df['session_id']
    # grab the targets
    target = df.target.values
    del df['target']

    # get the column index of the categorical columns
    categorical_ind = [k for k, v in enumerate(df.columns) if v in cat_fts]

    logger.info(f'Start training with cv and params:\n{params}')
    for fold, (trn_ind, val_ind) in enumerate(folds.split(splits, splits)):
        if split_mode == 'target':
            x_trn, y_trn = df.iloc[trn_ind], target[trn_ind]
            x_val, y_val = df.iloc[val_ind], target[val_ind]
            base_trn, base_val = base.iloc[trn_ind].copy(), base.iloc[val_ind].copy()
        else:
            trn_mask = df_sids.isin(sids[trn_ind])
            x_trn, y_trn = df[trn_mask], target[trn_mask]
            x_val, y_val = df[~trn_mask], target[~trn_mask]
            base_trn, base_val = base.loc[trn_mask].copy(), base.loc[~trn_mask].copy()
            # since the split index was based on only ids, we need to give it the actual index for oof
            val_ind = np.where(~trn_mask)

        lgb_trn = lgb.Dataset(x_trn, label=y_trn, categorical_feature=categorical_ind, free_raw_data=False)
        lgb_val = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_ind, free_raw_data=False)

        # train model
        clf = lgb.train(params,
                        lgb_trn,
                        valid_sets=[lgb_trn, lgb_val],
                        valid_names=['train', 'test'],
                        num_boost_round=200,
                        early_stopping_rounds=200,
                        verbose_eval=100)

        logger.info('Done training, generating predictions for oof, train and val for evaluation purpose')
        # assign prediction
        trn_rped =clf.predict(x_trn.values)
        val_pred = clf.predict(x_val.values)

        oof.loc[oof.index[val_ind], 'prediction'] = val_pred
        base_trn['prediction'] = trn_rped
        base_val['prediction'] = val_pred

        # compute mrr
        logger.info('Computing train mrr')
        trn_mrr = mean_reciprocal_rank(base_trn)
        logger.info('Computing val mrr')
        val_mrr = mean_reciprocal_rank(base_val)

        logger.info('Getting feature importance')
        # get feature importance
        imp_df = pd.DataFrame()
        imp_df['feature_importance'] = clf.feature_importance(importance_type='gain', iteration=clf.best_iteration)
        imp_df['features'] = x_trn.columns
        plot_imp_lgb(imp_df, f'lgb{fold}', mrr={'train': trn_mrr, 'val': val_mrr})

        logger.info('Done feature imp')
        break

    logger.info('Computing oof mrr')
    mean_reciprocal_rank(oof)
    logger.info('Done all')


def mean_reciprocal_rank(predictions):
    df = predictions.sort_values(by=['session_id', 'prediction'],
                                 ascending=[True, False]).reset_index(drop=True)
    pred_transposed = df.groupby('session_id')['impression'].apply(list).reset_index(name='recommendations')
    logger.debug(f'pred_transposed shape: {pred_transposed.shape}')
    df_reference = df[df['target'] != 0]
    logger.debug(f'df_reference shape: {df_reference.shape}')
    # assert len(pred_transposed) == len(df_reference), 'recommendations length and the target reference should be same length'
    combined = pd.merge(df_reference, pred_transposed, on='session_id')
    logger.debug(f'combined shape: {combined.shape}')
    # assert len(combined) == len(df_reference), 'combined reference and recommendations should match'

    def find_rank(x):
        t = x.impression
        ps = x.recommendations
        if t in ps:
            return 1/(ps.index(t)+1)
        else:
            return np.nan

    combined['mrr'] = combined.apply(find_rank, axis=1)
    mrr = combined['mrr'].mean()
    logger.info(f"MRR: {mrr}")
    return mrr


def run():
    RS = 42
    data_source = 'train'
    nrows = 10000000
    # nrows = None

    df = combine_inputs(data_source=data_source, nrows=nrows, reduce_memory_size=True, recompute=False)
    reduce_numeric_mem_usage(df)

    model = 'lgb'
    logger.info(f'Training with {model.upper()}')
    if model == 'lgb':
        params = {'objective': 'binary',
                       'metric': ['binary_logloss', 'auc'],
                       'seed': RS ,
                       'verbose': -1,
                       # hyper params
                       'num_leaves': 20,
                       'max_depth': 40,
                       # 'max_bin': 400,
                       # 'min_split_gain': 2.0,
                       # 'min_child_weight': 0.1,
                       # 'min_child_samples': 50,
                       'subsample': 0.8,
                       'subsample_freq': 5,
                       'colsample_bytree': 0.1,
                       'reg_alpha': 0.0,
                       'reg_lambda': 0.0}
        train_lgb(params, df=df, split_mode='sids', n_splits=5)
    else:
        device = 'GPU' if check_gpu() else 'CPU'
        logger.info(f'Using device: {device}')
        params = {'iterations': 1000,
                  'learning_rate': 0.02,
                  'depth': 8,
                  'task_type': device,
                  'loss_function': 'Logloss',
                  'custom_metric': ['Logloss', 'Accuracy', 'AUC', 'Precision', 'Recall'],
                  'eval_metric': 'Logloss'}
        if device == 'CPU':
            params['rsm'] = 0.5

        train_cat(params, df=df, split_mode='sids', n_splits=5)


if __name__ == '__main__':
    run()
