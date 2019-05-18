import pandas as pd
import subprocess
import os
import warnings
import multiprocessing
import logging
from datetime import datetime as dt
from collections import namedtuple


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


def get_data_path():
    path_dict = {'data_path': './data',
                 'cache_path': './cache',
                 'sub_path': './subs',
                 'model_path': './models',
                 'plot_path': './plots',
                 'log_path': './logs',
                 'tf_logs': './logs/tf_logs'}
    for _, v in path_dict.items():
        check_dir(v)
    FilePath = namedtuple('FilePath', list(path_dict.keys()))
    return FilePath(**path_dict)


Filepath = get_data_path()


def get_logger(name):
    logger_path = Filepath.log_path
    check_dir(logger_path)

    # add logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    if not logger.handlers:
        # create a file handler
        current_time = dt.now().strftime('%m-%d')
        file_handler = logging.FileHandler(os.path.join(logger_path, f'{current_time}.log'))
        file_handler.setLevel(logging.DEBUG)
        # create a logging format
        formats = '[%(asctime)s - %(name)s-%(lineno)d - %(funcName)s - %(levelname)s] %(message)s'
        file_formatter = logging.Formatter(formats, '%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)

        # console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.DEBUG)
        c_formatter = logging.Formatter(formats, '%m-%d %H:%M:%S')
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


def load_data(data_soruce, nrows=None, verbose=False, **kwargs):
    """
    Load csv files as dataframe
    :param data_soruce: str, train or test
    :param data_path: directory path where data sits
    :param nrows: number of rows to have
    :param verbose: boolean, print memory usage
    :param kwargs:
    :return: dataframe
    """
    ntrain = 15932993  # '15,932,993'
    ntest = 3782336
    data_path = Filepath.data_path
    # read
    df = pd.read_csv(os.path.join(data_path, data_soruce) + '.csv', nrows=nrows, **kwargs)
    if nrows is not None:
        total_rows = ntrain if data_soruce == 'train' else ntest
        # discard the last session as it could possibly be cut off by the nrows selection
        last_sid = df['session_id'].iloc[-1]
        df = df[df['session_id'] != last_sid].reset_index(drop=True)
        logger.info(f'Loading {data_soruce} using {nrows:,} rows ({len(df):,} trimmed) '
                    f'which is {len(df)/total_rows*100:.2f}% out of total {data_soruce} data')

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

