import os
import numpy as np
from functools import partial
import pandas as pd
from utils import get_logger, get_data_path


logger = get_logger('clean_session')
Filepath = get_data_path()


# 0)
def remove_duplicates(df):
    # find duplicates except steps
    df.sort_values(by=['user_id', 'session_id', 'timestamp', 'step'],
                   ascending=[True, True, True, True],
                   inplace=True)
    logger.info(f'Before dropping duplicates df shape: ({df.shape[0]:,}, {df.shape[1]})')
    cols = [c for c in df.columns if c != 'step']
    df = df.drop_duplicates(subset=cols, keep='last').reset_index(drop=True)
    logger.info(f'After dropping duplicates df shape: ({df.shape[0]:,}, {df.shape[1]})')
    return df


# 1) Cliping sessions up to last clickout (if there is clickout)
def clip_last_click(grp):
    check = (grp['action_type'].values == 'clickout item') & (pd.notna(grp['reference'].values))
    if check.sum() != 0:
        return grp.iloc[:np.argwhere(check)[-1][0]+1]
    else:
        return grp


# 2) Only select sessions that have a click out
def filter_clickout(grp, mode):
    # sessions has clickouts
    has_clickout = (grp['action_type'].values == 'clickout item').sum() != 0
    if mode == 'train':
        # last row has reference and it's not nan
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['impressions', 'reference', 'prices']].isna().sum() == 0))
    elif mode == 'test':
        # test should have the last reference as nan for clickout
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['reference']].isna()))
    else:
        raise ValueError('Invalid mode')
    return has_clickout & has_ref


def preprocess_sessions(df, mode, drop_duplicates=True, save=True, recompute=False):
    filepath = Filepath.nn_cache_path
    filename = os.path.join(filepath, f'preprocessed_{mode}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing file: {filename}')
        df = pd.read_parquet(filename)
    else:
        if drop_duplicates:
            logger.info('Dropping duplicates')
            df = remove_duplicates(df)
        logger.info('Cliping session dataframe up to last click out (if there is clickout)')
        df = df.groupby('session_id').apply(clip_last_click).reset_index(drop=True)
        logger.info(f'There are rares cases the reference is nan, shape: {df.shape}')
        # there are rares cases (seen in test set e.g. session_id=cbe3752713eee) the reference is nan
        df = df[(df['reference'].notna()) | (df['action_type'] != 'clickout item')].reset_index(drop=True)

        logger.info(f'After removing nan reference rows, shape: {df.shape}')
        logger.info('Filtering out sessions without clickouts, reference, or clickout is nan')
        logger.info(f'{mode} length before filtering: {len(df):,}')
        filter_func = partial(filter_clickout, mode=mode)
        valid_clicked = df.groupby('session_id').apply(filter_func)
        click_session_ids = valid_clicked[valid_clicked].index

        # filter
        df = df[df.session_id.isin(click_session_ids)].reset_index(drop=True)
        logger.info(f'{mode} length after filtering: {len(df):,}')
        if save:
            logger.info(f'Saving {filename}')
            df.to_parquet(filename)
    return df
