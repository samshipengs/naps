import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import get_data_path


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
