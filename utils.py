import numpy as np
import pandas as pd


def pshape(df):
    print(f'df len: {df.shape[0]:,}')


def load_data(data_soruce, nrows=None, **kwargs):
    data_path = './data/'
    return pd.read_csv(data_path+data_soruce+'.csv', nrows=nrows, **kwargs)


# some custom funcs used in agggregation
def mean_dwell_time(x):
    if len(x) == 1:
        return 0
    else:
        return np.mean(np.diff(np.sort(x)))


def var_dwell_time(x):
    if len(x) == 1:
        return 0
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