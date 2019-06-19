import os
import re
import time
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime as dt
from ast import literal_eval

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from sklearn.model_selection import KFold, ShuffleSplit
import xgboost as xgb

from create_model_inputs import create_model_inputs, CATEGORICAL_COLUMNS
from utils import get_logger, get_data_path, check_gpu, ignore_warnings
from plots import plot_hist, confusion_matrix, plot_imp_lgb, compute_shap_multi_class

# ignore_warnings()

logger = get_logger('train_xgb')
Filepath = get_data_path()
# random splitting for cross-validation
RS = 42


def compute_mrr(y_pred, y_true):
    pred_label = np.where(np.argsort(y_pred)[:, ::-1] == y_true.values.reshape(-1, 1))[1]
    return np.mean(1 / (pred_label + 1))


def evalmetric(preds, dtrain):
    labels = dtrain.get_label()
    pred_label = np.where(np.argsort(preds)[:, ::-1] == labels.reshape(-1, 1))[1]
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'mrr', 1 - np.mean(1 / (pred_label + 1))


def expand(df, col):
    """
    Expand a column of list to a list of columns and drop the original column
    :param df:
    :param col: column of df to expand on
    :return:
    """
    n_cols = len(df[col].iloc[0])
    expand_cols = [f'{col}_{i}' for i in range(n_cols)]
    df[expand_cols] = pd.DataFrame(df[col].values.tolist(), index=df.index)
    df.drop(col, axis=1, inplace=True)
    return df


def train(train_inputs, params, n_fold=5, test_fraction=0.15, only_last=False, feature_importance=True,
          retrain=False, verbose=True):
    # path to where model is saved
    model_path = Filepath.model_path

    # specify some columns that we do not want in training
    cf_cols = [i for i in train_inputs.columns if 'current_filters' in i]
    price_cols = [i for i in train_inputs.columns if re.match(r'prices_\d', i)]
    drop_cols = cf_cols + price_cols  # + ['country', 'platform']
    # drop cf col for now
    train_inputs.drop(drop_cols, axis=1, inplace=True)
    logger.debug(f'train columns: {train_inputs.columns}')
    # if only use the last row of train_inputs to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_inputs = train_inputs.groupby('session_id').last().reset_index(drop=False)

    train_inputs['last_action_type'] = train_inputs['last_action_type'].apply(lambda x: np.eye(11, dtype=int)[x][1:])
    # convert last_action_type to one hot
    expand(train_inputs, 'last_action_type')

    # grab unique session ids and use this to split, so that train_inputs with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_inputs['session_id'].unique()

    # kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    ss = ShuffleSplit(n_splits=n_fold, test_size=test_fraction, random_state=RS)

    # record classifiers and mrr each training
    clfs = []
    mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(ss.split(unique_session_ids)):
        t1 = time.time()
        logger.info(f'Training fold {fold}: train ids len={len(trn_ind):,} | val ids len={len(val_ind):,}')
        # get session_id used for train
        trn_ids = unique_session_ids[trn_ind]
        trn_mask = train_inputs['session_id'].isin(trn_ids)
        logger.info(f'Training fold {fold}: train len={trn_mask.sum():,} | val len={(~trn_mask).sum():,}')

        x_trn, x_val = (train_inputs[trn_mask].reset_index(drop=True),
                        train_inputs[~trn_mask].reset_index(drop=True))

        if not only_last:
            # for validation only last row is needed
            x_val = x_val.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn, y_val = x_trn['target'].values, x_val['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        x_val.drop(['session_id', 'target'], axis=1, inplace=True)

        # get categorical index
        # cat_ind = [k for k, v in enumerate(x_trn.columns) if v in CATEGORICAL_COLUMNS]
        # =====================================================================================

        # lgb_trn_data = lgb.Dataset(x_trn, label=y_trn, free_raw_data=False)
        # lgb_val_data = lgb.Dataset(x_val, label=y_val, free_raw_data=False)
        dtrain = xgb.DMatrix(x_trn, label=y_trn)
        dval = xgb.DMatrix(x_val, label=y_val)

        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'lgb_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f"Loading model from existing '{model_filename}'")
            # parameters not required.
            clf = xgb.Booster(model_file=model_filename)
        else:
            # train model
            logger.info('Starts training')
            clf = xgb.train(params,
                            dtrain,
                            num_boost_round=100,
                            early_stopping_rounds=10,
                            feval=evalmetric,
                            evals=[(dtrain, 'train'), (dval, 'val')])

            if feature_importance:
                logger.info('Compute feature importance')
                # grab feature importance
                imp_df = pd.DataFrame()
                imp_df['feature_importance'] = clf.feature_importance(importance_type='gain',
                                                                      iteration=clf.best_iteration)
                imp_df['features'] = x_trn.columns
                plot_imp_lgb(imp_df, fold)
                compute_shap_multi_class(clf, x_val, x_val.columns, f'lgb_shap_{fold}')

            clf.save_model(model_filename)

        # make prediction
        x_trn = train_inputs[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        y_trn = x_trn['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        dtrain = xgb.DMatrix(x_trn, label=y_trn)

        trn_pred = clf.predict(dtrain)

        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = clf.predict(dval)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(clf)
        mrrs.append((trn_mrr, val_mrr))

        if verbose:
            logger.info(f'Done training {fold}, took: {(time.time()-t1)/60:.2f} mins')

    logger.info(f'Total cv time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


if __name__ == '__main__':
    setup = {'nrows': 1000000,
             'recompute_train': False,
             'add_test': False,
             'only_last': True,
             'retrain': True,
             'recompute_test': False}

    params = {'eta': 0.05,
              'gamma': 0,
              'max_depth': 5,
              'subsample': 0.9,
              'colsample_bytree': 0.4,
              'alpha': 5,
              'grow_policy': 'depthwise',  # depthwise, lossguide
              'objective': 'multi:softprob',
              'num_class': 25,
              'nthread': 11,
              'silent': 0,
              }

    if params['grow_policy'] == 'lossguide':
        params['max_leaves'] = 12

    if check_gpu():
        params['tree_method'] = 'gpu_hist'
        logger.info('Using GPU')
    else:
        logger.info('Using CPU')

    logger.info(f"\nSetup\n{'=' * 20}\n{pprint.pformat(setup)}\n{'=' * 20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], padding_value=np.nan,
                                       add_test=setup['add_test'], recompute=setup['recompute_train'])

    logger.info('Training')
    # train the model
    logger.info(f"\nParams\n{'=' * 20}\n{pprint.pformat(params)}\n{'=' * 20}")
    models, mrrs = train(train_inputs, params=params, only_last=setup['only_last'], feature_importance=False,
                         retrain=setup['retrain'])
    train_mrr = np.mean([mrr[0] for mrr in mrrs])
    val_mrr = np.mean([mrr[1] for mrr in mrrs])
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
        test_pred = clf.predict(test_inputs)
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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'lgb_sub_{current_time}_{train_mrr:.4f}_{val_mrr:.4f}.csv'),
                    index=False)
    logger.info('Done all')
