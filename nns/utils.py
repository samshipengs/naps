import numpy as np
import pandas as pd
import subprocess
import sys
import os
import matplotlib.pyplot as plt
import warnings
import multiprocessing
import logging
from datetime import datetime as dt


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


class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)


def get_logger(name):
    logger_path = './loggers'
    check_dir(logger_path)

    # add logging
    logger = logging.getLogger(name)
    sys.stdout = LoggerWriter(logger.debug)
    sys.stderr = LoggerWriter(logger.warning)
    logger.setLevel(logging.INFO)
    # https://stackoverflow.com/questions/6729268/log-messages-appearing-twice-with-python-logging
    if not logger.handlers:
        # create a file handler
        current_time = dt.now().strftime('%m-%d')
        file_handler = logging.FileHandler(os.path.join(logger_path, f'{current_time}.log'))
        file_handler.setLevel(logging.INFO)
        # create a logging format
        file_formatter = logging.Formatter(('[%(asctime)s - %(name)s-%(lineno)d - %(funcName)s - %(levelname)s ] '
                                           '%(message)s'),
                                           '%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)

        # console handler
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_formatter = logging.Formatter('[%(asctime)s - %(name)s-%(lineno)d - %(funcName)s - %(levelname)s] %(message)s',
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


def load_data(data_soruce, data_path='../data/', nrows=None, verbose=False, **kwargs):
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
    if nrows is not None:
        if data_soruce == 'train':
            load_per = nrows/ntrain
        else:
            load_per = nrows/ntest
        logger.info(f'Loading {data_soruce} using {nrows:,} rows which is {load_per*100:.2f}% out of total train data')
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

