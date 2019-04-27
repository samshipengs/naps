import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from clean_session import preprocess_sessions
from reduce_memory import reduce_numeric_mem_usage
from session_features import compute_session_fts
from hotel2vec import hotel2vec
from manual_encoding import action_encoding, click_view_encoding, meta_encoding
from utils import load_data, get_logger, check_dir, check_gpu


logger = get_logger('data_pipeline')


def explode(df):
    impressions = df['impressions'].values.tolist()
    prices = df['prices'].values.tolist()

    rs = [len(r) for r in impressions]
    # locations
    inds = np.concatenate([np.arange(i, dtype=int) for i in rs])
    # relative locations
    rel_inds = np.concatenate([np.arange(i) / i for i in rs])

    # the rest cols
    rest_cols = [c for c in df.columns if c not in ['impressions', 'prices']]
    rest_arr = np.repeat(df[rest_cols].values, rs, axis=0)
    # create dataframe to host the exploded
    exploded = pd.DataFrame(np.column_stack((rest_arr, np.concatenate(impressions), np.concatenate(prices))),
                            columns=rest_cols + ['impression', 'price'])
    exploded['price'] = exploded['price'].astype(int)
    exploded['impression_loc'] = inds
    exploded['rel_impression_loc'] = rel_inds
    exploded['impression'] = exploded['impression'].astype(int)
    exploded['reference'] = exploded['reference'].astype(int)
    exploded['step'] = exploded['step'].astype(int)

    return exploded


def compute_diff(df, grp, cols):
    diff = (df.set_index('session_id')[cols] - grp[cols].mean()).reset_index(drop=True)
    diff.columns = [f'{c}_diff' for c in diff.columns]
    return pd.concat([df, diff], axis=1)


def combine_inputs(data_source='train', nrows=None, reduce_memory_size=False, recompute=False):
    logger.info(f'Start data processing pipeline, first load raw {data_source} data')

    filepath = './cache'
    check_dir(filepath)
    filename = os.path.join(filepath, f'{data_source}_combined_inputs.snappy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load {data_source} from existing {filename}')
        logger.warning('Since this is a reload, it may not reflect the latest change in settings')
        df = pd.read_parquet(filename)
        return df

    df = load_data('train', nrows=nrows)
    df = preprocess_sessions(df, data_source='train', rd=True)
    drop_cols = ['user_id', 'timestamp', 'current_filters']
    df = df.drop(drop_cols, axis=1)
    logger.info(f'Selecting only the last row of each session, current len:{df.shape[0]:,}')
    df = df.groupby('session_id').last().reset_index()
    logger.info(f'After selecting only the last row, len:{df.shape[0]:,}')

    # split to list
    df['impressions'] = df.impressions.str.split('|')
    df['prices'] = df.prices.str.split('|')

    logger.info('Exploding on impressions and prices')
    df = explode(df)
    logger.info(f'After exploding, shape: ({df.shape[0]:,}, {df.shape[1]})')

    # 1) all the manual encodings
    ae = action_encoding()
    ae_cols = [c for c in ae.columns if c != 'reference']
    # reduce memory
    if reduce_memory_size:
        reduce_numeric_mem_usage(ae, ae_cols)
    assert df['impression'].dtype == ae['reference'].dtype, 'dtype not matching'
    df = pd.merge(df.set_index('impression'), ae.set_index('reference'), left_index=True, right_index=True)
    del ae
    gc.collect()
    df.index.name = 'impression'
    df.reset_index(inplace=True)
    logger.info(f'After adding action encodings, shape: ({df.shape[0]:,}, {df.shape[1]})')

    # 2) the hotel2vec encodings
    hv = hotel2vec()
    hv_cols = [c for c in hv.columns if c != 'item_id']
    if reduce_memory_size:
        reduce_numeric_mem_usage(hv, hv_cols)
    assert df['impression'].dtype == hv['item_id'].dtype, 'dtype not matching'
    df = pd.merge(df.set_index('impression'), hv.set_index('item_id'), left_index=True, right_index=True)
    del hv
    gc.collect()
    df.index.name = 'impression'
    df.reset_index(inplace=True)
    logger.info(f'After adding hotelvec, shape: ({df.shape[0]:,}, {df.shape[1]})')

    # 3) click view
    cv = click_view_encoding()
    cv_cols = [c for c in cv.columns if c != 'item_id']
    if reduce_memory_size:
        reduce_numeric_mem_usage(cv, cv_cols)
    assert df['impression'].dtype == cv['item_id'].dtype, 'dtype not matching'
    df = pd.merge(df.set_index('impression'), cv.set_index('item_id'), left_index=True, right_index=True)
    del cv
    df.index.name = 'impression'
    df.reset_index(inplace=True)
    logger.info(f'After adding clickview, shape: ({df.shape[0]:,}, {df.shape[1]})')

    # 4) meta
    meta = meta_encoding()
    meta_cols = [c for c in meta.columns if c != 'item_id']
    if reduce_memory_size:
        reduce_numeric_mem_usage(meta, meta_cols)
    assert df['impression'].dtype == meta['item_id'].dtype, 'dtype not matching'
    df = pd.merge(df.set_index('impression'), meta.set_index('item_id'), left_index=True, right_index=True)
    del meta
    df.index.name = 'impression'
    df.reset_index(inplace=True)
    logger.info(f'After adding meta, shape: ({df.shape[0]:,}, {df.shape[1]})')

    # groupby
    grp = df.groupby('session_id')
    # compute the relative difference
    df = compute_diff(df, grp, ['price'])
    df = compute_diff(df, grp, ae_cols)
    df = compute_diff(df, grp, hv_cols)
    df = compute_diff(df, grp, cv_cols)
    df = compute_diff(df, grp, meta_cols)

    # create target
    df['target'] = (df['reference'] == df['impression']).astype(int)
    del df['reference']

    logger.info(f'Done combing data, shape: ({df.shape[0]:,}, {df.shape[1]})')
    df.to_parquet(filename)
    logger.info(f'Done saving {filename}')

    return df


if __name__ == '__main__':
    data_source = 'train'
    nrows = 1000000
    # nrows = None
    df = combine_inputs(data_source=data_source, nrows=nrows, recompute=True)