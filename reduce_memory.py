import pandas as pd
import numpy as np
import time
import argparse
from utils import get_logger


logger = get_logger('reduce_memory')

def reduce_object_mem_usage(df, mode='mapping', interested_cols=None):
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Memory usage before optimization is: {:.2f} MB'.format(start_mem))

    if interested_cols is None:
        o_cols = df.dtypes[df.dtypes == 'O'].index
    else:
        o_cols = df[interested_cols].dtypes[df[interested_cols].dtypes == 'O'].index

    if mode == 'mapping':
        logger.info('Reducing with int mapping')
        for col in o_cols:
            # mapping
            mapping = {v: k for k, v in enumerate(df[col].unique())}
            max_ = df[col].nunique()
            if max_ < np.iinfo(np.int8).max:
                df[col] = df[col].map(mapping).astype(np.int8)
            elif max_ < np.iinfo(np.int16).max:
                df[col] = df[col].map(mapping).astype(np.int16)
            elif max_ < np.iinfo(np.int32).max:
                df[col] = df[col].map(mapping).astype(np.int32)
            elif max_ < np.iinfo(np.int64).max:
                df[col] = df[col].map(mapping).astype(np.int64)
    else:
        logger.info('Reducing with dtype category')
        for col in o_cols:
            # change to categorical type
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))


# reduce memory on numeric types
def reduce_numeric_mem_usage(df, interested_cols=None):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Memory usage before optimization is: {:.2f} MB'.format(start_mem))
    columns = df.columns if interested_cols is None else interested_cols
    for col in columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))


class Elasped:
    def __init__(self, time_int):
        self.initial_time = time_int

    def timer(self, msg):
        diff = np.round((time.time() - self.initial_time) / 60)
        logger.info('{} | time elapsed: {}mins'.format(msg, diff))
        return diff


def main(file_name):
    logger.info('loading data')
    t_int = time.time()
    p = Elasped(t_int)
    data_sample = pd.read_csv(file_name)
    p.timer('Done loading')

    initial_memory = data_sample.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Original memeory footprint: {0:.2f} MB'.format(initial_memory))

    logger.info('Reducing object type mem usage')
    reduce_object_mem_usage(data_sample)
    p.timer('Done')

    logger.info('Reducing numeric type mem usage')
    reduce_numeric_mem_usage(data_sample)
    p.timer('Done')

    end_memory = data_sample.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info('Reduced by {:.2f}%'.format(100 * (initial_memory - end_memory) / initial_memory))
    save_file = 'reduced.h5'
    data_sample.to_hdf(save_file, key='data')
    p.timer('Done saving file to {}'.format(save_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='File name')
    args = parser.parse_args()
    main(args.file_name)