import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from clean_session import preprocess_sessions
# from reduce_memory import reduce_numeric_mem_usage
# from session_features import compute_session_fts
# from hotel2vec import hotel2vec
# from manual_encoding import action_encoding, click_view_encoding, meta_encoding
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
        # TODO
        df = pd.read_parquet('filename')
    logger.info(df['target'].value_counts())
    # categorize
    cat_fts = ['city', 'platform', 'device', 'action_type', 'impression']
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

    oof = df[['session_id', 'target']].copy()
    del df['session_id']

    # grab the targets
    target = df.target.values
    del df['target']

    categorical_ind = [k for k, v in enumerate(df.columns) if v in cat_fts]

    for trn_ind, val_ind in folds.split(splits, splits):
        if split_mode == 'target':
            x_trn, y_trn = df.iloc[trn_ind], target[trn_ind]
            x_val, y_val = df.iloc[val_ind], target[val_ind]
        else:
            trn_mask = df['session_id'].isin(sids[trn_ind])
            x_trn, y_trn = df[trn_mask], target[trn_mask]
            x_val, y_val = df[~trn_mask], target[~trn_mask]

        # train model
        clf = cat.CatBoostClassifier(**params)
        clf.fit(x_trn.values, y_trn,
                cat_features=categorical_ind,
                eval_set=(x_val.values, y_val),
                early_stopping_rounds=100,
                verbose=100,
                plot=False)
        # assign prediction to oof
        oof.loc[val_ind, 'prediction'] = clf.predict_proba(x_val.values)[:, 1]
        logger.info('Done training')
        logger.info('Getting feature importance for both train and val')
        # get feature importance
        trn_imp = clf.get_feature_importance(data=cat.Pool(data=x_trn, cat_features=categorical_ind),
                                             prettified=True)
        val_imp = clf.get_feature_importance(data=cat.Pool(data=x_val, cat_features=categorical_ind),
                                             prettified=True)
        plot_imp(trn_imp, 'train')
        plot_imp(val_imp, 'val')
        logger.info('Done feature imp')

        # compute mrr
        logger.info('Computing mrr')
        mean_reciprocal_rank(oof.loc[val_ind, 'prediction'])

        break


def mean_reciprocal_rank(predictions):
    df = predictions.sort_values(by=['session_id', 'prediction'], ascending=[True, False]).reset_index(drop=True)
    pred_transposed = df.groupby('session_id').apply(list).reset_index(name='predictions')
    df_reference = df[df['target'] != 0]
    combined = pd.merge(df_reference, pred_transposed, on='session_id')
    def find_rank(x):
        t = x.target
        ps = x.predictions
        if t in ps:
            return (ps.index(t)+1)/len(ps)
        else:
            return np.nan
    combined['mrr'] = combined.apply(find_rank, axis=1)
    logger.info(f"MRR: {combined['mrr'].mean()}")
    return combined


def plot_imp(data, fold_, plot_n=15):
    check_dir('./imps')
    imp = pd.DataFrame.from_records(data)
    imp.to_csv(f'./imps/{fold_}.csv', index=False)
    imp.columns = ['features', 'feature_importance']
    imp_des = imp.sort_values(by='feature_importance', ascending=False)
    imp_asc = imp.sort_values(by='feature_importance', ascending=True)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    imp_des[:plot_n].plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    imp_asc[:plot_n].plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig('./imps/{}.png'.format(fold_))



def run():
    data_source = 'train'
    nrows = 1000000
    # nrows = None
    df = combine_inputs(data_source=data_source, nrows=nrows, recompute=True)

    device = 'GPU' if check_gpu() else 'CPU'
    params = {'iterations': 1000,
              'learning_rate': 0.02,
              'depth': 8,
              'task_type': device,
              'loss_function': 'MultiClass',
              'eval_metric': 'Accuracy'}
    train(params, df=df, split_mode='target', n_splits=5)


if __name__ == '__main__':
    run()
