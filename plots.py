import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_data_path, check_dir


Filepath = get_data_path()


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
    check_dir('./imps')
    imp = pd.DataFrame.from_records(data)
    imp.to_csv(f'./imps/{fold_}.csv', index=False)
    imp.columns = ['features', 'feature_importance']
    imp_des = imp.sort_values(by='feature_importance', ascending=False)
    imp_asc = imp.sort_values(by='feature_importance', ascending=True)

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    # axes[0].set_title(f"trn_mrr={np.round(mrr['train'], 4)} - val_mrr={np.round(mrr['val'], 4)}")
    imp_des[:plot_n].plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    imp_asc[:plot_n].plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig('./imps/{}.png'.format(fold_))


def plot_imp_lgb(imp_df, fold_, plot_n=15):
    """
    This funciton is to plot the feature importance for top plot_n features
    :param imp_df: feature importance dataframe
    :param fold_: current cross validation fold
    :param plot_n: top n features to plot on feature importanc graph
    """
    check_dir('./imps')
    imp_df.sort_values(by='feature_importance', ascending=False).to_csv('./imps/{}.csv'.format(fold_))
    imp_des = imp_df.sort_values(by='feature_importance', ascending=False)[:plot_n]
    imp_asc = imp_df.sort_values(by='feature_importance', ascending=True)[:plot_n]

    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    _ = imp_des.plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    _ = imp_asc.plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig('./imps/{}.png'.format(fold_))
    plt.gcf().clear()
