import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import pandas as pd
from utils import get_data_path, check_dir


Filepath = get_data_path()
IMP_PATH = Filepath.imp_path


def plot_hist(pred_label, true_label, name):
    _ = plt.hist(pred_label, bins=50, label=f'{name}_pred', alpha=0.7)
    _ = plt.hist(true_label, bins=50, label=f'{name} label', alpha=0.7)
    _ = plt.legend()
    plt.savefig(os.path.join(Filepath.plot_path, f'{name}_hist.png'))
    plt.gcf().clear()


def confusion_matrix(y_pred, y_true, name, normalize='row', level=0, log_scale=False):
    compare = pd.DataFrame({'prediction': y_pred, 'y_true': y_true})
    counts = compare.groupby('y_true')['prediction'].value_counts()
    mat = counts.unstack(level=level)
    mat.fillna(0, inplace=True)

    if normalize == 'row':
        row_sum = mat.sum(axis=1)
        mat = mat.div(row_sum, axis=0)
        log_scale = False
    elif normalize == 'column':
        col_sum = mat.sum(axis=0)
        mat = mat.div(col_sum, axis=1)
        log_scale = False
    # plot
    fig = plt.figure(figsize=(35, 10))
    ax = fig.add_subplot(111)
    if log_scale:
        cax = ax.matshow(np.log1p(mat), interpolation='nearest')  # , cmap='coolwarm')#, aspect='auto')
    else:
        cax = ax.matshow(mat, interpolation='nearest')  # , cmap='coolwarm')#, aspect='auto')
    fig.colorbar(cax)
    ax.set_xlabel(f'{mat.columns.name}')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(f'{mat.index.name}')

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(list(mat.columns.astype(str)), rotation=90)
    ax.set_yticks(np.arange(mat.shape[0]))
    _ = ax.set_yticklabels(list(mat.index.astype(str)))
    plt.savefig(os.path.join(Filepath.plot_path, f'{name}_confusion_matrix.png'))
    plt.gcf().clear()


# def plot_imp_cat(data, fold_, mrr, plot_n=15):
def plot_imp_cat(data, fold_, plot_n=15):
    imp = pd.DataFrame.from_records(data)
    imp.to_csv(os.path.join(IMP_PATH, f'cat_{fold_}.csv'), index=False)
    imp.columns = ['features', 'feature_importance']
    imp_des = imp.sort_values(by='feature_importance', ascending=False)
    imp_asc = imp.sort_values(by='feature_importance', ascending=True)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    # axes[0].set_title(f"trn_mrr={np.round(mrr['train'], 4)} - val_mrr={np.round(mrr['val'], 4)}")
    imp_des[:plot_n].plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    imp_asc[:plot_n].plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig(os.path.join(IMP_PATH, f'cat_{fold_}.png'))


def plot_imp_lgb(imp_df, fold_, plot_n=15):
    """
    This funciton is to plot the feature importance for top plot_n features
    :param imp_df: feature importance dataframe
    :param fold_: current cross validation fold
    :param plot_n: top n features to plot on feature importanc graph
    """
    imp_df.sort_values(by='feature_importance', ascending=False).to_csv(os.path.join(IMP_PATH, f'lgb_{fold_}.csv'))
    imp_des = imp_df.sort_values(by='feature_importance', ascending=False)[:plot_n]
    imp_asc = imp_df.sort_values(by='feature_importance', ascending=True)[:plot_n]

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    _ = imp_des.plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    _ = imp_asc.plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig(os.path.join(IMP_PATH, f'lgb_{fold_}.png'))
    plt.gcf().clear()


def plot_shap_imp(imp, fold_, plot_n=20):
    """
    Plot most and least important features and save the complete (mean) feature importances to csv.
    :param imp: feature importance dataframe
    :param fold_: number of cv
    :param plot_n: number of most and least important features
    :return:
    """
    # first save all to csv
    imp.sort_values(by='feature_importance', ascending=False).to_csv(os.path.join(IMP_PATH, f'shap_{fold_}.csv'),
                                                                     index=False)

    # select plot_n number of most and least important features to plot
    imp_des = imp.sort_values(by='feature_importance', ascending=False)[:plot_n]
    imp_asc = imp.sort_values(by='feature_importance', ascending=True)[:plot_n]

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    imp_des.plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    imp_asc.plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig(os.path.join(IMP_PATH, f'shap_{fold_}.png'))
    plt.gcf().clear()


def compute_shap(model, x_val, cv_i):
    """
    Compute shap value
    :param model: trained model
    :param xtrain: complete training data
    :param val_ind: index of the validation set, shap is computed on validation set
    :param cv_i: number of the cv
    :return: array of column names that is in decreasing order of shap value, used for selecting
    """
    # xtrain_ = xtrain.copy()
    # init tree shap
    explainer = shap.TreeExplainer(model)
    # # grab the validation data
    # val = xtrain_.iloc[val_ind].reset_index(drop=True)
    # compute shap values on validation set
    shap_values_val = explainer.shap_values(x_val)
    # compute the average (abs) shap values for each features
    avg_shap = np.mean(np.abs(shap_values_val), axis=0)
    # write the result to a df
    imp = pd.DataFrame()
    imp['features'] = x_val.columns
    imp['feature_importance'] = avg_shap
    # plot the shap value
    plot_shap_imp(imp, cv_i, plot_n=15)
    # sort the feature imp in descending order
    imp.sort_values(by='feature_importance', ascending=False, inplace=True)
    # # get the column names in decreasing order of shap value
    # import_val = val.columns[np.argsort(avg_shap)][::-1]
    return imp


# def plot_shap_imp(val, shap_values, fold_, plot_n=15):
#     imp = pd.DataFrame()
#     imp['features'] = val.columns
#     imp['feature_importance'] = shap_values
#     imp_des = imp.sort_values(by='feature_importance', ascending=False)
#     imp_asc = imp.sort_values(by='feature_importance', ascending=True)
#
#     check_dir('./imps')
#     fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
#     imp_des[:plot_n].plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
#     imp_asc[:plot_n].plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
#     plt.tight_layout()
#     fig.savefig('./imps/shap_{}.png'.format(fold_))