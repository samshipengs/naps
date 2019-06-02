import os
import datetime
import numpy as np
from functools import partial
import pandas as pd
from utils import load_data, get_logger, get_data_path, flogger


logger = get_logger('preprocessing')
Filepath = get_data_path()

# CONVERT_ACTION_TYPE = True
USE_COLS = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters', 'reference', 'impressions', 'prices']


# 0)
def remove_duplicates(df):
    """
    Drop rows with every columns the same except step
    :param df: dataframe, (almost) raw df from load_data
    :return: dataframe
    """
    logger.info('Dropping exactly same rows (except step)')
    # find duplicates except steps
    df = (df
          .sort_values(by=['user_id', 'session_id', 'timestamp', 'step'], ascending=[True, True, True, True])
          .reset_index(drop=True))
    logger.info(f'Before dropping duplicates df shape: ({df.shape[0]:,}, {df.shape[1]})')

    # get columns used to drop duplicates
    cols = [c for c in df.columns if c != 'step']
    df = df.drop_duplicates(subset=cols, keep='last').reset_index(drop=True)
    logger.info(f'After dropping duplicates df shape: ({df.shape[0]:,}, {df.shape[1]})')
    return df


# 1) Clipping sessions up to last click-out (if there is click-out)
def clip_up_to_last_click(grp, mode):
    """
    Clip session records up to last click-out if the session has click-outs, otherwise remove the session
    :param grp: dataframe associated with a group key from groupby
    # :param click_out: str or int, representation of click-out, it's either 0 or 'clickout item' depends on encoding
    :param mode: str, 'train' or 'test'
    :return: clipped dataframe
    """
    if mode == 'train':
        # in train mode, we go up to the last non-null reference click-outs, this is necessary especially when we try
        # to make use of test data (except the last row)
        check = (grp['action_type'].values == 'clickout item') & (pd.notna(grp['reference'].values))
    elif mode == 'test':
        check = (grp['action_type'].values == 'clickout item') & (pd.isna(grp['reference'].values))
    else:
        raise ValueError('Invalid mode')

    if np.sum(check) != 0:
        return grp.iloc[:np.argwhere(check)[-1][0]+1]

    else:
        # else drop the df
        return pd.DataFrame()


# 2) Only select sessions that have a click out
def filter_and_check(grp, mode):
    """
    Select only valid sessions, e.g. the last click-out reference, impressions or prices shall not be nan.
    For test set, last click-out reference should be nan
    :param grp: dataframe associated with a group key from groupby
    :param mode: str, 'train' or 'test'
    :return: a boolean mask that indicates the valid sessions
    """
    # sessions has clickouts
    has_clickout = (grp['action_type'].values == 'clickout item').sum() != 0
    if mode == 'train':
        # last row has reference and it's not nan
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['impressions', 'reference', 'prices']].isna().sum() == 0))
    elif mode == 'test':
        # test should have the last reference as nan for click-out
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (pd.isna(grp.iloc[-1]['reference'])))
    else:
        raise ValueError('Invalid mode')
    return has_clickout & has_ref


def basic_preprocess_sessions(df, mode, nrows, drop_duplicates=True, save=True, recompute=False):
    """
    Trigger the whole basic processing steps
    :param df: raw dataframe
    :param mode: 'train' or 'test'
    :param nrows: number of rows that was used to the input df
    :param drop_duplicates: whether drop duplicated rows (i.e. all same except step)
    :param save:
    :param recompute:
    :return: basic cleaned dataframe
    """
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, f'basic_preprocessed_{mode}_{nrows}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing basic pre-processing file: {filename}')
        df = pd.read_parquet(filename)
    else:
        # 0) drop duplicates
        if drop_duplicates:
            df = remove_duplicates(df)
        # 1) clipping to last click-out if any
        logger.info('Clipping session dataframe up to last click out (if there is clickout)')
        # clickout = 0 if CONVERT_ACTION_TYPE else 'clickout item'
        # clip_up_to_last_click_ = partial(clip_up_to_last_click, click_out=clickout, mode=mode)
        clip_up_to_last_click_ = partial(clip_up_to_last_click, mode=mode)
        df = df.groupby('session_id').apply(clip_up_to_last_click_).reset_index(drop=True)

        # 2) select only valid sessions
        logger.info(f'Select valid sessions, {mode} length before selecting: {len(df):,}')
        filter_and_check_ = partial(filter_and_check, mode=mode)
        valid_clicked = df.groupby('session_id').apply(filter_and_check_)
        # if one right, there should be no invalid sessions at this point
        invalid_session_count = (~valid_clicked).sum()
        assert invalid_session_count == 0, f'There are {invalid_session_count} invalid sessions'
        # grab the invalid session ids and filter
        click_session_ids = valid_clicked[valid_clicked].index
        df = df[df['session_id'].isin(click_session_ids)].reset_index(drop=True)
        logger.info(f'{mode} length after selecting: {len(df):,}')
        if save:
            logger.info(f'Saving {filename}')
            df.to_parquet(filename)
    return df


def create_action_type_mapping(recompute=False):
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, 'action_types_mapping.npy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load action_types mapping from existing: {filename}')
        action_type2natural = np.load(filename).item()
        n_unique_actions = len(action_type2natural)
    else:
        # hardcoded, each action will be represented as the index in following list
        # e.g. clickout item -> 0
        actions = ['clickout item', 'search for poi', 'interaction item image',
                   'interaction item info', 'interaction item deals',
                   'search for destination', 'filter selection',
                   'interaction item rating', 'search for item',
                   'change of sort order']
        action_type2natural = {v: k for k, v in enumerate(actions)}
        n_unique_actions = len(actions)
        np.save(filename, action_type2natural)
    return action_type2natural, n_unique_actions


def preprocess_data(mode, nrows=None, add_test=True, recompute=False):
    nrows_str = 'all' if nrows is None else nrows
    add_test_str = 'test_added' if add_test else 'no_test_added'
    filename = os.path.join(Filepath.gbm_cache_path, f'{mode}_{nrows_str}_{add_test_str}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
        # first load data
        if mode == 'train':
            df = load_data(mode, nrows=nrows)
            if add_test:
                logger.info(f'Add available test data, number of rows of current raw data is: {len(df):,}')
                df_test = load_data('test', nrows=nrows)
                df = pd.concat([df, df_test], axis=0, ignore_index=True)
                logger.info(f'After test data added, number of rows of becomes: {len(df):,}')
        else:
            if type(nrows) == int:
                logger.warning(f'Full test data always gets loaded when in test mode, nrows={nrows} is not used')
            df = load_data(mode)
            flogger(df, 'Load test with only what ids needed for submissions')
            # load  only the test that we need to submit
            required_sids = load_data('submission_popular', usecols=['session_id'])['session_id'].unique()
            df = df[df['session_id'].isin(required_sids)].reset_index(drop=True)
            logger.info(f'Rows of raw data that is required for test submission is: {len(df):,} '
                        f'for {len(required_sids):,} unique sessions')

        flogger(df, f'raw {mode}')

        # if CONVERT_ACTION_TYPE:
        #     logger.info('Converting action_types to int (natural number)')
        #     action_type2natural, _ = create_action_type_mapping(recompute=False)
        #     df['action_type'] = df['action_type'].map(action_type2natural)

        # basic pre-process data i.e. dropping duplicates, only take sessions with clicks and clip to last click out
        df = basic_preprocess_sessions(df, mode=mode, nrows=nrows, drop_duplicates=True, save=True, recompute=recompute)

        # get time and select columns that get used
        df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.datetime.utcfromtimestamp(ts))
        df = df[USE_COLS]

        logger.info('Sort df by session_id, timestamp, step')
        df = df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
        df.to_parquet(filename)
    return df