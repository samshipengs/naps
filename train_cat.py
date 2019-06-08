import os
import time
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ast import literal_eval

from sklearn.model_selection import KFold
import catboost as cat

from create_model_inputs import create_model_inputs
from utils import get_logger, get_data_path, check_gpu
from plots import plot_hist, confusion_matrix, plot_imp_cat


logger = get_logger('train_model')
Filepath = get_data_path()
# random splitting for cross-validation
RS = 42


def train(train_inputs, params, only_last=False, retrain=False):
    # path to where model is saved
    model_path = Filepath.model_path
    cf_cols = [c for c in train_inputs.columns if 'current_filters' in c]
    # drop cf col for now
    train_inputs.drop(cf_cols, axis=1, inplace=True)

    # if only use the last row of train_inputs to train
    if only_last:
        logger.info('Training ONLY with last row')
        train_inputs = train_inputs.groupby('session_id').last().reset_index(drop=False)

    # grab unique session ids and use this to split, so that train_inputs with same session_id do not spread to both
    # train and valid
    unique_session_ids = train_inputs['session_id'].unique()

    kf = KFold(n_splits=5, shuffle=True, random_state=RS)

    # record classifiers and mrr each training
    clfs = []
    mrrs = []
    t_init = time.time()
    for fold, (trn_ind, val_ind) in enumerate(kf.split(unique_session_ids)):
        logger.info(f'Training fold {fold}: train ids len={len(trn_ind):,} | val ids len={len(val_ind):,}')
        # get session_id used for train
        trn_ids = unique_session_ids[trn_ind]
        trn_mask = train_inputs['session_id'].isin(trn_ids)
        logger.info(f'Training fold {fold}: train len={trn_mask.sum():,} | val ids len={(~trn_mask).sum():,}')

        x_trn, x_val = (train_inputs[trn_mask].reset_index(drop=True),
                        train_inputs[~trn_mask].reset_index(drop=True))

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
        # create model
        model_filename = os.path.join(model_path, f'cat_cv{fold}.model')
        if os.path.isfile(model_filename) and not retrain:
            logger.info(f'Loading model from existing {model_filename}')
            # parameters not required.
            clf = cat.CatBoostClassifier()
            clf.load_model(model_filename)
        else:
            # train model
            clf = cat.CatBoostClassifier(**params)
            clf.fit(x_trn, y_trn,
                    cat_features=cat_ind,
                    eval_set=(x_val, y_val),
                    early_stopping_rounds=100,
                    verbose=100,
                    plot=False)
            trn_imp = clf.get_feature_importance(prettified=True,
                                                 type='FeatureImportance')
            plot_imp_cat(trn_imp, fold)
            clf.save_model(model_filename)

        # make prediction
        x_trn = train_inputs[trn_mask].reset_index(drop=True)
        x_trn = x_trn.groupby('session_id').last().reset_index(drop=False)
        y_trn = x_trn['target'].values
        x_trn.drop(['session_id', 'target'], axis=1, inplace=True)

        trn_pred = clf.predict_proba(x_trn)
        trn_pred_label = np.where(np.argsort(trn_pred)[:, ::-1] == y_trn.reshape(-1, 1))[1]
        plot_hist(trn_pred_label, y_trn, 'train')
        confusion_matrix(trn_pred_label, y_trn, 'train', normalize=None, level=0, log_scale=True)
        trn_mrr = np.mean(1 / (trn_pred_label + 1))

        val_pred = clf.predict_proba(x_val)
        val_pred_label = np.where(np.argsort(val_pred)[:, ::-1] == y_val.reshape(-1, 1))[1]
        plot_hist(val_pred_label, y_val, 'validation')
        confusion_matrix(val_pred_label, y_val, 'val', normalize=None, level=0, log_scale=True)
        val_mrr = np.mean(1 / (val_pred_label + 1))
        logger.info(f'train mrr: {trn_mrr:.4f} | val mrr: {val_mrr:.4f}')

        clfs.append(clf)
        mrrs.append((trn_mrr, val_mrr))

    logger.info(f'Total time took: {(time.time()-t_init)/60:.2f} mins')
    return clfs, mrrs


if __name__ == '__main__':
    setup = {'nrows': None,
             'recompute_train': True,
             'add_test': False,
             'only_last': True,
             'retrain': True,
             'recompute_test': True}

    device = 'GPU' if check_gpu() else 'CPU'
    params = {'loss_function': 'MultiClass',
              'custom_metric': ['MultiClass', 'Accuracy'],
              'eval_metric': 'MultiClass',
              'iterations': 10000,
              'learning_rate': 0.02,
              # 'depth': 8,
              # 'min_data_in_leaf': 2,
              'task_type': device}

    logger.info(f"\nSetup\n{'='*20}\n{setup}\n{'='*20}")
    logger.info(f"\nParams\n{'='*20}\n{params}\n{'='*20}")

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
        test_pred = clf.predict_proba(test_inputs)
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
    test_sub.to_csv(os.path.join(Filepath.sub_path, f'cat_sub_{current_time}_{train_mrr:.4f}_{val_mrr:.4f}.csv'), index=False)
    logger.info('Done all')

