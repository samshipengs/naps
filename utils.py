import numpy as np
import pandas as pd
import subprocess
import os
from time import time
import matplotlib.pyplot as plt
import warnings
import multiprocessing
import logging


def get_logger(name):
    # add logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # create a file handler
    file_handler = logging.FileHandler('logger.log')
    file_handler.setLevel(logging.INFO)
    # create a logging format
    file_formatter = logging.Formatter('[%(asctime)s - %(name)8s - %(funcName)s - %(levelname)s] %(message)s',
                                       '%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    # add the handlers to the logger
    logger.addHandler(file_handler)

    # console handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_formatter = logging.Formatter('[%(asctime)s - %(name)s - %(funcName)s - %(levelname)s] %(message)s',
                                    '%m-%d %H:%M:%S')
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)
    return logger


logger = get_logger('utils')


def ignore_warnings():
    """
    Ignore warnings
    """
    logger.warning('WARNING IS BEING DISABLED! INCLUDING: PerformanceWarning, FutureWarning, UserWarning')
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


def pshape(df, name='df'):
    """
    Print df shape with , separated
    :param df:
    :return:
    """
    print(f'[>>>>>] {name} shape: ({df.shape[0]:,}, {df.shape[1]})')


def load_data(data_soruce, data_path='./data/', nrows=None, verbose=False, **kwargs):
    """
    Load csv files as dataframe
    :param data_soruce: str, train or test
    :param data_path: directory path where data sits
    :param nrows: number of rows to have
    :param verbose: boolean, print memory usage
    :param kwargs:
    :return: dataframe
    """
    assert  data_soruce in ['train', 'test']
    ntrain = 15932993
    ntest = 3782336
    if nrows is not None:
        if data_soruce == 'train':
            load_per = nrows/ntrain
        else:
            load_per = nrows/ntest
        logger.info(f'Loading {data_soruce} using {nrows:,} rows which is {load_per:.2f}% out of total train data')
    # read
    df = pd.read_csv(os.path.join(data_path, data_soruce) + '.csv', nrows=nrows, **kwargs)

    if verbose:
        if (nrows is None) and (data_soruce in ['train', 'test']):
            logger.warning('Getting memory usage would take a while (~ 1min)')
        logger.info(f'Memory usage: {df.memory_usage(deep=True).sum()/1024**2:.2f} mb')
    return df


def check_gpu():
    """
    Check whether gpu is available or not
    :return:
    """
    try:
        n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        if n > 0:
            return True
        else:
            return False
    except:
        return False


def get_cpu_count(stable=True):
    ncpu = multiprocessing.cpu_count()
    if stable:
        ncpu -= 1
    logger.info(f'[number of cpu count: {ncpu}]')
    return ncpu

def check_dir(dirs):
    """
    Create a or a list of directories
    :param dirs:
    :return:
    """
    if type(dirs) == list:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
    else:
        if not os.path.exists(dirs):
            os.makedirs(dirs)


def plot_imp_cat(data, fold_, mrr, plot_n=15):
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


def plot_imp_lgb(imp_df, fold_, mrr, plot_n=15):
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
    axes[0].set_title(f"trn_mrr={np.round(mrr['train'], 4)} - val_mrr={np.round(mrr['val'], 4)}")
    _ = imp_des.plot(x='features', y='feature_importance', ax=axes[0], kind='barh', grid=True)
    _ = imp_asc.plot(x='features', y='feature_importance', ax=axes[1], kind='barh', grid=True)
    plt.tight_layout()
    fig.savefig('./imps/{}.png'.format(fold_))
    plt.gcf().clear()