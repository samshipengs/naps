import os
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
import lightgbm as lgb

from create_model_inputs import create_model_inputs
from utils import get_logger, get_data_path
from plots import plot_hist, confusion_matrix, plot_imp_lgb


logger = get_logger('train_lgb')
Filepath = get_data_path()
# random splitting for cross-validation
RS = 42


def compute_mrr(y_pred, y_true):
    pred_label = np.where(np.argsort(y_pred)[:, ::-1] == y_true.values.reshape(-1, 1))[1]
    return np.mean(1 / (pred_label + 1))


def train(train_inputs, params, only_last=False, retrain=False):
    # path to where model is saved
    model_path = Filepath.model_path

    # specify some columns that we do not want in training
    cf_cols = [c for c in train_inputs.columns if 'current_filters' in c]
    drop_cols = cf_cols  # + ['country', 'platform']
    # drop cf col for now
    train_inputs.drop(drop_cols, axis=1, inplace=True)
    logger.info(f'train columns: {train_inputs.columns}')
    # if only use the last row of train_inputs to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_inputs = train_inputs.groupby('session_id').last().reset_index(drop=False)

    # grab unique session ids and use this to split, so that train_inputs with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_inputs['session_id'].unique()

    # kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    ss = ShuffleSplit(n_splits=5, test_size=0.15, random_state=RS)

    # record classifiers and mrr each training
    clfs = []
    mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(ss.split(unique_session_ids)):
        logger.info(f'Training fold {fold}: train ids len={len(trn_ind):,} | val ids len={len(val_ind):,}')
        # get session_id used for train
        trn_ids = unique_session_ids[trn_ind]
        trn_mask = train_inputs['session_id'].isin(trn_ids)
        logger.info(f'Training fold {fold}: train len={trn_mask.sum():,} | val ids len={(~trn_mask).sum():,}')

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
        cat_cols = ['country', 'device', 'platform', 'fs', 'cs']
        cat_ind = [k for k, v in enumerate(x_trn.columns) if v in cat_cols]
        # =====================================================================================

        lgb_trn_data = lgb.Dataset(x_trn, label=y_trn, free_raw_data=False)
        lgb_val_data = lgb.Dataset(x_val, label=y_val, free_raw_data=False)
        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'lgb_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f'Loading model from existing {model_filename}')
            # parameters not required.
            clf = lgb.Booster(model_file=model_filename)
        else:
            # train model
            clf = lgb.train(params,
                            lgb_trn_data,
                            valid_sets=[lgb_trn_data, lgb_val_data],
                            valid_names=['train', 'val'],
                            categorical_feature=cat_ind,
                            # init_model=lgb.Booster(model_file=model_filename),
                            verbose_eval=100)
            # grab feature importances
            imp_df = pd.DataFrame()
            imp_df['feature_importance'] = clf.feature_importance(importance_type='gain',
                                                                  iteration=clf.best_iteration)
            imp_df['features'] = x_trn.columns
            plot_imp_lgb(imp_df, fold)

            clf.save_model(model_filename)

        # make prediction
        x_trn = train_inputs[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        y_trn = x_trn['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)

        trn_pred = clf.predict(x_trn)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = clf.predict(x_val)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(clf)
        mrrs.append((trn_mrr, val_mrr))

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


def lgb_tuning(xtrain, n_searches=100):
    """
    Tuning hyperparams through Bayesian opt
    :param xtrain: input train dataframe
    :param n_searches: number of calls to make in the gaussian process
    :return: best params dictionary and best score
    """
    # specify the range of hyperparams values to go through
    dim_num_leaves = Integer(low=10, high=40, name='num_leaves')
    dim_max_depth = Integer(low=5, high=40, name='max_depth')
    dim_max_bin = Integer(low=50, high=400, name='max_bin')
    dim_min_split_gain = Real(low=0, high=10, name='min_split_gain')
    dim_min_child_weight = Real(low=0, high=10, name='min_child_weight')
    dim_min_child_samples = Integer(low=10, high=50, name='min_child_samples')
    dim_subsample = Real(low=0.1, high=0.99, name='subsample')
    dim_subsample_freq = Integer(low=1, high=10, name='subsample_freq')
    dim_colsample_bytree = Real(low=0.1, high=0.99, name='colsample_bytree')
    dim_reg_alpha = Real(low=0, high=10, name='reg_alpha')
    dim_reg_lambda = Real(low=0, high=10, name='reg_lambda')

    dimensions = [dim_num_leaves,
                  dim_max_depth,
                  dim_max_bin,
                  dim_min_split_gain,
                  dim_min_child_weight,
                  dim_min_child_samples,
                  dim_subsample,
                  dim_subsample_freq,
                  dim_colsample_bytree,
                  dim_reg_alpha,
                  dim_reg_lambda]

    @use_named_args(dimensions=dimensions)
    def fitness(**params):
        # train(train_inputs, params, only_last=False, retrain=False):
        eval_func = partial(train,
                            train_inputs=xtrain,
                            only_last=False,
                            retrain=False)
        # modeling from CV
        _, mrrs = eval_func(params)
        return -np.mean(mrrs)

    # search
    t1 = time.time()
    # n_cpu = multiprocessing.cpu_count() - 1
    search_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                acq_func='gp_hedge',  # Expected Improvement.
                                n_calls=n_searches,
                                # n_jobs=n_cpu,
                                verbose=True)
    t2 = time.time()

    best_score = search_result.fun
    logger.info('Params tuning took: {0:.2f}mins'.format((t2 - t1) / 60))
    logger.info('Best score = {0:.4f}'.format(best_score))
    logger.info('Best params found:\n' + '=' * 20)
    # get the best params
    best_params = {}
    for k, v in enumerate(dimensions):
        best_params[v.name] = search_result.x[k]
        print(f'{v.name}: {search_result.x[k]}')
    best_params_name = os.path.join(Filepath.opt_path, f'best_params_{best_score:.5f}.npy')
    np.save(best_params_name, best_params)
    logger.info('Saved best_params to {}'.format(best_params_name))
    # plot gp convergence
    plot_convergence(search_result)
    plt.savefig(os.path.join(Filepath.opt_path, 'gp_convergence.png'))
    return best_params, best_score


if __name__ == '__main__':
    setup = {'nrows': None,
             'recompute_train': False,
             'add_test': False,
             'only_last': False,
             'retrain': True,
             'recompute_test': False}

    params = {'boosting': 'gbdt',  # gbdt, dart, goss
              'max_depth': -1,
              'num_leaves': 12,
              'feature_fraction': 0.8,
              'num_boost_round': 5000,
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

    logger.info(f"\nSetup\n{'=' * 20}\n{pprint.pformat(setup)}\n{'=' * 20}")
    logger.info(f"\nParams\n{'=' * 20}\n{pprint.pformat(params)}\n{'=' * 20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], padding_value=np.nan,
                                       add_test=setup['add_test'], recompute=setup['recompute_train'])
    # train the model
    models, mrrs = train(train_inputs, params=params, only_last=setup['only_last'], retrain=setup['retrain'])
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

