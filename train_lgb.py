import os
import time
import pandas as pd
import numpy as np
from datetime import datetime as dt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import lightgbm as lgb

from create_model_inputs import create_model_inputs, click_view_encoding
from utils import get_logger, get_data_path, ignore_warnings
from plots import plot_hist, confusion_matrix, plot_imp_lgb

ignore_warnings()

logger = get_logger('train_lgb')
Filepath = get_data_path()


def cv_encode(df, mapping):
    imp_cols = [f'imp_{i}' for i in range(25)]
    for c in imp_cols:
        df[c] = df[c].map(mapping)


# def compute_mrr_lgb(y_pred_flat, dtrain):
#     y_pred = y_pred_flat.reshape(-1, 25)
#     y_true = dtrain.get_label()
#     # print(y_pred.shape, y_true.shape)
#     pred_label = np.where(np.argsort(y_pred)[:, ::-1] == y_true.values.reshape(-1, 1))[1]
#     mrr = np.mean(1 / (pred_label + 1))
#     return 'mrr', mrr, True


def compute_mrr(y_pred, y_true):
    pred_label = np.where(np.argsort(y_pred)[:, ::-1] == y_true.values.reshape(-1, 1))[1]
    return np.mean(1 / (pred_label + 1))


def train(train_inputs, params, add_cv_encoding=False, retrain=False, continue_train=False):
    cache_path = Filepath.gbm_cache_path
    model_path = Filepath.model_path

    targets = train_inputs['target']
    train_inputs.drop('target', axis=1, inplace=True)

    # skf = StratifiedKFold(n_splits=6)
    sss = StratifiedShuffleSplit(n_splits=6, test_size=0.15, random_state=42)

    clfs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(sss.split(targets, targets)):
        logger.info(f'Training fold {fold}: train len={len(trn_ind):,} | val len={len(val_ind):,}')
        x_trn, x_val = train_inputs.iloc[trn_ind].reset_index(drop=True), train_inputs.iloc[val_ind].reset_index(
            drop=True)
        y_trn, y_val = targets.iloc[trn_ind], targets.iloc[val_ind]
        # cv encoding
        if add_cv_encoding:
            sids_trn = x_trn['session_id'].unique()
            logger.info('Add click-view/impression encodings')
            cv_encoding = click_view_encoding(sids_trn, fold, m=100, nrows=None, recompute=False)

            cv_encode(x_trn, cv_encoding)
            cv_encode(x_val, cv_encoding)
            x_trn.drop('session_id', axis=1, inplace=True)
            x_val.drop('session_id', axis=1, inplace=True)
        lgb_trn_data = lgb.Dataset(x_trn, label=y_trn, free_raw_data=False)
        lgb_val_data = lgb.Dataset(x_val, label=y_val, free_raw_data=False)

        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'lgb_classifier{fold}')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f'Loading model from existing {model_filename}')
            clf = lgb.Booster(model_file=model_filename)
        else:
            if continue_train:
                # train model
                assert os.path.isfile(model_filename), f'{model_filename} does not exist!'
                clf = lgb.train(params,
                                lgb_trn_data,
                                valid_sets=[lgb_trn_data, lgb_val_data],
                                valid_names=['train', 'val'],
                                init_model=lgb.Booster(model_file=model_filename),
                                verbose_eval=100)
            else:
                # train model
                clf = lgb.train(params,
                                lgb_trn_data,
                                valid_sets=[lgb_trn_data, lgb_val_data],
                                valid_names=['train', 'val'],
                                # feval=compute_mrr_lgb,
                                verbose_eval=100)
            # grab feature importances
            imp_df = pd.DataFrame()
            imp_df['feature_importance'] = clf.feature_importance(importance_type='gain',
                                                                  iteration=clf.best_iteration)
            imp_df['features'] = x_trn.columns
            plot_imp_lgb(imp_df, fold)

            clf.save_model(model_filename)

        # make prediction
        trn_pred = clf.predict(x_trn)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.values.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = clf.predict(x_val)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.values.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(clf)

    logger.info(f'Total time took: {(time.time() - t_init) / 60:.2f} mins')
    return clfs


if __name__ == '__main__':
    setup = {'nrows': None,
             'add_cv_encoding': False,
             'recompute_train': False,
             'retrain': True,
             'continue_train': True,
             'recompute_test': False}

    params = {'boosting': 'gbdt',  # gbdt, dart, goss
              'max_depth': 5,
              'num_leaves': 6,
              'feature_fraction': 0.9,
              'num_boost_round': 3000,
              'early_stopping_rounds': 100,
              'learning_rate': 0.01,
              'objective': 'multiclass',
              'num_class': 25,
              'metric': ['multi_logloss'],
              'verbose': -1,
              'seed': 42,
              }
    if params['boosting'] != 'goss':
        params['bagging_fraction'] = 0.9
        params['bagging_freq'] = 1

    logger.info(f"\nSetup\n{'=' * 20}\n{setup}\n{'=' * 20}")
    logger.info(f"\nParams\n{'=' * 20}\n{params}\n{'=' * 20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], add_cv_encoding=setup['add_cv_encoding'],
                                       recompute=setup['recompute_train'])
    # train the model
    models = train(train_inputs, params=params, add_cv_encoding=setup['add_cv_encoding'], retrain=setup['retrain'],
                   continue_train=setup['continue_train'])
    # get the test inputs
    test_inputs = create_model_inputs(mode='test', nrows=setup['nrows'], add_cv_encoding=setup['add_cv_encoding'],
                                      recompute=setup['recompute_test'])
    # make predictions on test
    logger.info('Load test sub csv')
    test_sub = pd.read_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'))
    sub_popular = pd.read_csv(os.path.join(Filepath.data_path, 'submission_popular.csv'))
    sub_columns = sub_popular.columns

    # filter away the 0 padding and join list recs to string
    def create_recs(recs):
        return ' '.join([i for i in recs if i != 0])


    test_predictions = []
    for c, clf in enumerate(models):
        test_sub_m = test_sub.copy()
        logger.info(f'Generating predictions from model {c}')
        if setup['add_cv_encoding']:
            cv_encoding = click_view_encoding(sids=None, fold=c, m=5, nrows=None, recompute=False)
            cv_encode(test_inputs, cv_encoding)
        test_pred = clf.predict(test_inputs)
        test_predictions.append(test_pred)

    logger.info('Generating submission by averaging cv predictions')
    test_predictions = np.array(test_predictions).mean(axis=0)
    test_pred_label = np.argsort(test_predictions)[:, ::-1]
    np.save(os.path.join(Filepath.sub_path, f'test_pred_label.npy'), test_pred_label)

    # pad to 25
    test_sub['impressions'] = test_sub['impressions'].str.split('|')
    print(test_sub['impressions'].str.len().describe())
    test_sub['impressions'] = test_sub['impressions'].apply(lambda x: np.pad(x, (0, 25 - len(x)), mode='constant'))
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
    current_time = dt.now().strftime('%m-%d')
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'lgb_sub_{current_time}.csv'), index=False)
    logger.info('Done all')

