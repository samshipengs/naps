"""
Train model and perform necessary steps such as convert categorical column to natural number and provide categorical
index for lightgbm and catboost
"""
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, get_logger, check_dir, check_gpu, get_logger
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


def train(params, df=None, split_mode='target', n_splits=5):
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
        plot_imp(trn_imp, fold, mrr={'train': trn_mrr, 'val': val_mrr})
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


def plot_imp(data, fold_, mrr, plot_n=15):
    check_dir('./imps')
    imp = pd.DataFrame.from_records(data)
    imp.to_csv(f'./imps/{fold_}.csv', index=False)
    imp.columns = ['features', 'feature_importance']
    imp_des = imp.sort_values(by='feature_importance', ascending=False)
    imp_asc = imp.sort_values(by='feature_importance', ascending=True)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    axes[0].set_title(f"trn_mrr={np.round(mrr['train'], 4)} - val_mrr={np.round(mrr['val'], 4)}")
    imp_des[:plot_n].plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    imp_asc[:plot_n].plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig('./imps/{}.png'.format(fold_))


def run():
    data_source = 'train'
    ntrain = 15932993
    nrows = 10000000
    # nrows = None
    if nrows is not None:
        logger.info(f'Training using {nrows:,} rows which is {nrows/ntrain:.2f}% out of total train data')
    df = combine_inputs(data_source=data_source, nrows=nrows, reduce_memory_size=True, recompute=True)

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

    train(params, df=df, split_mode='sids', n_splits=5)


if __name__ == '__main__':
    run()
