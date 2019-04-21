import numpy as np
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import warnings


def ignore_warnings():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)


def pshape(df):
    print(f'df len: {df.shape[0]:,}')


def load_data(data_soruce, data_path='./data/', nrows=None, verbose=False, **kwargs):
    df = pd.read_csv(data_path+data_soruce+'.csv', nrows=nrows, **kwargs)
    if verbose:
        if (nrows is None) and (data_soruce in ['train', 'test']):
            print('Warning: getting memory usage would take a while (~ 1min)')
        print(f'Memory usage: {df.memory_usage(deep=True).sum()/1024**2:.2f} mb')
    return df


# some custom funcs used in agggregation
def mean_dwell_time(x):
    if len(x) == 1:
        return np.nan
    else:
        return np.mean(np.diff(np.sort(x)))


def var_dwell_time(x):
    if len(x) == 1:
        return np.nan
    else:
        return np.var(np.diff(np.sort(x)))


def get_first(x):
    return x.iloc[0]


def get_last(x):
    return x.iloc[-1]


def n_clickouts(x):
    return (x == 'clickout item').sum()


def click_rel_pos_avg(x):
    return np.mean(np.argwhere((x == 'clickout item'))) / len(x)


def ptp(x):
    return x.max() - x.min()


def check_gpu():
    try:
        n = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        if n > 0:
            return True
        else:
            return False
    except:
        return False


def check_dir(dirs):
    if type(dirs) == list:
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)
    else:
        if not os.path.exists(dirs):
            os.makedirs(dirs)


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