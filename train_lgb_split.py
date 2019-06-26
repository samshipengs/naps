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

from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
from sklearn.model_selection import KFold, ShuffleSplit
import lightgbm as lgb

from create_model_inputs import create_model_inputs, CATEGORICAL_COLUMNS
from utils import get_logger, get_data_path, check_gpu, ignore_warnings
from plots import plot_hist, confusion_matrix, plot_imp_lgb, compute_shap_multi_class

ignore_warnings()

logger = get_logger('train_lgb')
Filepath = get_data_path()
# random splitting for cross-validation
RS = 42


def compute_mrr(y_pred, y_true):
    pred_label = np.where(np.argsort(y_pred)[:, ::-1] == y_true.reshape(-1, 1))[1]
    return np.mean(1 / (pred_label + 1))


def log_median(df, col):
    df[col] = np.log((1+df[col])/(1+np.median(df[col])))


def lgb_preprocess(df):
    # specify some columns that we do not want in training
    cf_cols = [i for i in df.columns if 'current_filters' in i]
    price_cols = [i for i in df.columns if re.match(r'prices_\d', i)]

    drop_cols = cf_cols + price_cols + ['country', 'platform', 'impressions_str']
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

    logger.info(f'COLUMNS:\n{list(df.columns)}')

    filename = 'train_inputs_lgb.snappy'
    df.to_parquet(os.path.join(Filepath.cache_path, filename), index=False)
    return df


def train(train_df, params, n_fold=5, test_fraction=0.15, only_last=False, feature_importance=True,
          retrain=False, verbose=True):
    # path to where model is saved
    model_path = Filepath.model_path

    # a bit processing
    train_df = lgb_preprocess(train_df)

    train_cols = train_df.columns
    logger.info(f'Columns used for training:\n{train_cols.values}')
    used_categorical_cols = train_cols[train_cols.isin(CATEGORICAL_COLUMNS)]
    logger.info(f'Categorical columns in training:\n{used_categorical_cols.values}')

    # grab unique session ids and use this to split, so that train_df with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_df['session_id'].unique()

    # kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    ss = ShuffleSplit(n_splits=n_fold, test_size=test_fraction, random_state=RS)

    # add sizes info
    train_df['length'] = train_df.groupby('session_id')['session_id'].transform('size')

    # record classifiers and mrr each training
    clfs = []
    mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(ss.split(unique_session_ids)):
        t1 = time.time()
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

        logger.info(f'Train has {trn_ones_mask.sum()} ones session and {(~trn_ones_mask).sum()} more')

        x_trn_ones, x_trn_more = x_trn[trn_ones_mask].reset_index(drop=True), x_trn[~trn_ones_mask].reset_index(drop=True)
        x_val_ones, x_val_more = x_val[val_ones_mask].reset_index(drop=True), x_val[~val_ones_mask].reset_index(drop=True),

        # for validation only last row is needed
        # x_val_ones = x_val_ones.groupby('session_id').last().reset_index(drop=False)  # this is not needed
        x_val_more = x_val_more.groupby('session_id').last().reset_index(drop=False)

        # get target
        y_trn_ones, y_val_ones = x_trn_ones['target'].values, x_val_ones['target'].values
        y_trn_more, y_val_more = x_trn_more['target'].values, x_val_more['target'].values

        remove_ones_cols = [c for c in x_trn.columns if 'prev' in c]
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
        lgb_trn_data_ones = lgb.Dataset(x_trn_ones, label=y_trn_ones, free_raw_data=False)
        lgb_val_data_ones = lgb.Dataset(x_val_ones, label=y_val_ones, free_raw_data=False)

        lgb_trn_data_more = lgb.Dataset(x_trn_more, label=y_trn_more, free_raw_data=False)
        lgb_val_data_more = lgb.Dataset(x_val_more, label=y_val_more, free_raw_data=False)
        # =====================================================================================
        # create model
        model_filename = os.path.join(model_path, f'lgb_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f"Loading model from existing '{model_filename}'")
            # parameters not required.
            clf = lgb.Booster(model_file=model_filename)
        else:
            # train model
            params_ones = {'boosting': 'gbdt',  # gbdt, dart, goss
                           'num_boost_round': 500,
                           'learning_rate': 0.01,
                           'early_stopping_rounds': 50,
                           'num_class': 25,
                           'objective': 'multiclass',
                           'metric': ['multi_logloss'],
                           'verbose': -1,
                           'seed': 42,
                           'max_depth': 5,
                           'num_leaves': 8,
                           'feature_fraction': 1}

            clf_one = lgb.train(params_ones,
                                lgb_trn_data_ones,
                                valid_sets=[lgb_trn_data_ones, lgb_val_data_ones],
                                valid_names=['train', 'val'],
                                categorical_feature=cat_ind_ones,
                                # feval=lgb_mrr,
                                # init_model=lgb.Booster(model_file=model_filename),
                                verbose_eval=100)

            params_more = {'boosting': 'gbdt',  # gbdt, dart, goss
                           'num_boost_round': 500,
                           'learning_rate': 0.01,
                           'early_stopping_rounds': 100,
                           'num_class': 25,
                           'objective': 'multiclass',
                           'metric': ['multi_logloss'],
                           'verbose': -1,
                           'seed': 42,
                           'max_depth': 5,
                           'num_leaves': 10,
                           'feature_fraction': 0.9,
                           }

            clf_more = lgb.train(params_more,
                                 lgb_trn_data_more,
                                 valid_sets=[lgb_trn_data_more, lgb_val_data_more],
                                 valid_names=['train', 'val'],
                                 categorical_feature=cat_ind_more,
                                 # feval=lgb_mrr,
                                 # init_model=lgb.Booster(model_file=model_filename),
                                 verbose_eval=100)

            # if feature_importance:
            #     logger.info('Compute feature importance')
            #     # grab feature importance
            #     imp_df = pd.DataFrame()
            #     imp_df['feature_importance'] = clf.feature_importance(importance_type='gain',
            #                                                           iteration=clf.best_iteration)
            #     imp_df['features'] = x_trn.columns
            #     plot_imp_lgb(imp_df, fold)
            #     compute_shap_multi_class(clf, x_val, x_val.columns, f'lgb_shap_{fold}')
            #
            # clf.save_model(model_filename)

        # # make prediction
        # x_trn = train_df[trn_mask].reset_index(drop=True)
        # x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        # y_trn = x_trn['target'].values
        # x_trn.drop(['session_id', 'target'], axis=1, inplace=True)
        #
        # trn_pred = clf.predict(x_trn)
        # trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        # # plot_hist(trn_pred_label, y_trn, 'train')
        # # confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        # trn_mrr = np.mean(1 / (trn_pred_label + 1))
        # # save prediction
        # np.save(os.path.join(model_path, 'lgb_trn_0_pred.npy'), trn_pred)

        val_pred = np.concatenate((clf_one.predict(x_val_ones), clf_more.predict(x_val_more)), axis=0)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == np.concatenate((y_val_ones, y_val_more), axis=0).reshape(-1, 1))[1]
        # plot_hist(val_pred_label, y_val, 'validation')
        # confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        # logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')
        logger.info(f'val mrr: {val_mrr:.4f}')

        np.save(os.path.join(model_path, 'lgb_val_0_pred.npy'), val_pred)

        clfs.append(clf)
        mrrs.append((trn_mrr, val_mrr))

        if verbose:
            logger.info(f'Done training {fold}, took: {(time.time()-t1)/60:.2f} mins')

    logger.info(f'Total cv time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


def lgb_tuning(xtrain, base_params, n_searches=200):
    """
    Tuning hyperparams through Bayesian opt
    :param xtrain: input train dataframe
    :param n_searches: number of calls to make in the gaussian process
    :return: best params dictionary and best score
    """
    # specify the range of hyperparams values to go through
    dim_num_leaves = Integer(low=10, high=40, name='num_leaves')
    dim_max_depth = Integer(low=5, high=10, name='max_depth')
    # dim_max_bin = Integer(low=50, high=100, name='max_bin')
    # dim_min_split_gain = Real(low=0, high=10, name='min_split_gain')
    # dim_min_child_weight = Real(low=0, high=10, name='min_child_weight')
    # dim_min_child_samples = Integer(low=10, high=50, name='min_child_samples')
    # dim_subsample = Real(low=0.1, high=0.99, name='subsample')
    # dim_subsample_freq = Integer(low=1, high=10, name='subsample_freq')
    dim_colsample_bytree = Real(low=0.1, high=0.99, name='colsample_bytree')
    # dim_reg_alpha = Real(low=0, high=10, name='reg_alpha')
    # dim_reg_lambda = Real(low=0, high=10, name='reg_lambda')

    dimensions = [dim_num_leaves,
                  dim_max_depth,
                  # dim_max_bin,
                  # dim_min_split_gain,
                  # dim_min_child_weight,
                  # dim_min_child_samples,
                  # dim_subsample,
                  # dim_subsample_freq,
                  dim_colsample_bytree,
                  # dim_reg_alpha,
                  # dim_reg_lambda
                  ]

    @use_named_args(dimensions=dimensions)
    def fitness(**opt_params):
        logger.info(f'Experimenting with params:\n{opt_params}')
        hyper_params = opt_params
        # get base params
        for base_k, base_v in base_params.items():
            hyper_params[base_k] = base_v
        # train(train_inputs, params, only_last=False, retrain=False):
        eval_func = partial(train,
                            n_fold=2,
                            test_fraction=0.15,
                            train_inputs=xtrain,
                            only_last=False,
                            feature_importance=False,
                            retrain=True,
                            verbose=False)
        # modeling from CV
        _, cv_mrrs = eval_func(params=hyper_params)
        # only grab the validation mrr
        val_mrrs = [m[1] for m in cv_mrrs]
        return -np.mean(val_mrrs)

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
    logger.info(f"Saved best_params to '{best_params_name}'")
    # plot gp convergence
    plot_convergence(search_result)
    plt.savefig(os.path.join(Filepath.opt_path, 'gp_convergence.png'))
    return best_params, best_score


if __name__ == '__main__':
    setup = {'nrows': 1000000,
             'tuning': False,
             'recompute_train': False,
             'add_test': False,
             'only_last': False,
             'retrain': True,
             'recompute_test': True}

    base_params = {'boosting': 'gbdt',  # gbdt, dart, goss
                   'num_boost_round': 3000,
                   'learning_rate': 0.03,
                   'early_stopping_rounds': 100,
                   'num_class': 25,
                   'objective': 'multiclass',
                   'metric': ['multi_logloss'],
                   'verbose': -1,
                   'seed': 42
                   }

    params = {'max_depth': 5,
              'num_leaves': 10,
              'feature_fraction': 0.9,
              }

    if base_params['boosting'] != 'goss':
        params['bagging_fraction'] = 0.9
        params['bagging_freq'] = 1

    if check_gpu():
        params['device'] = 'gpu'
        params['max_bin'] = 15  # 63
        params['gpu_use_dp'] = False
        logger.info('Using GPU')
    else:
        logger.info('Using CPU')

    logger.info(f"\nSetup\n{'=' * 20}\n{pprint.pformat(setup)}\n{'=' * 20}")

    # first create training inputs
    train_inputs = create_model_inputs(mode='train', nrows=setup['nrows'], padding_value=np.nan,
                                       add_test=setup['add_test'], recompute=setup['recompute_train'])
    if setup['tuning']:
        logger.info('Tuning')
        logger.info(f"\nBase params\n{'=' * 20}\n{pprint.pformat(base_params)}\n{'=' * 20}")
        lgb_tuning(train_inputs, base_params=base_params)
    else:
        logger.info('Training')
        for k, v in base_params.items():
            params[k] = v
        # train the model
        logger.info(f"\nParams\n{'=' * 20}\n{pprint.pformat(params)}\n{'=' * 20}")
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
            np.save(os.path.join(Filepath.sub_path, 'lgb_test_0_pred.npy'), test_pred)
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

