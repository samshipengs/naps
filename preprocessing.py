import os
import datetime
import numpy as np
from functools import partial
import pandas as pd
from utils import load_data, get_logger, get_data_path, flogger


logger = get_logger('preprocessing')
Filepath = get_data_path()

# USE_COLS = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters', 'reference', 'impressions', 'prices',
#             'country', 'device', 'platform']
USE_COLS = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters',
            'reference', 'impressions', 'prices', 'device', 'country', 'device', 'platform']


# 0)
def remove_duplicates(df):
    """
    Drop rows with every columns the same except step
    :param df: dataframe, (almost) raw df from load_data
    :return: dataframe
    """
    logger.info('Dropping exactly same rows (except step). '
                f'Before dropping duplicates df shape: ({df.shape[0]:,}, {df.shape[1]})')
    # 1 (the quicker way does not work due to same session_id sessions are not necessarily side by side, and sort
    # may create more troubles)
    # sizes = df.groupby('session_id').size()
    # sids = df['session_id'].unique()
    # assert len(sids) == len(sizes)
    # sizes = sizes.reindex(index=sids)
    # df['step'] = np.concatenate([list(range(s)) for s in sizes])

    # correct steps
    def correct_steps(grp):
        grp['step'] = list(range(1, grp.shape[0] + 1))
        return grp

    df = df.groupby('session_id').apply(correct_steps)
    df['step'] = df['step'].astype('int')
    n_duplicates = df.duplicated(subset=['session_id', 'step']).sum()
    assert n_duplicates == 0, f'There should be 0 duplicates of session_id and step but found {n_duplicates}'

    # sort by session_id and steps
    df = df.sort_values(by=['session_id', 'step']).reset_index(drop=True)

    # 2
    unique_identifiers = ['session_id', 'action_type', 'reference']  # , 'impressions']
    for c in unique_identifiers:
        df[f'last_{c}'] = df[c].shift(1)
    # select mask
    select_mask = np.ones(df.shape[0], dtype=bool)
    for c in unique_identifiers:
        c_mask = df[c] == df[f'last_{c}']
        select_mask = select_mask & c_mask

    df = df[~select_mask].reset_index(drop=True)
    df.drop([f'last_{c}' for c in unique_identifiers], axis=1, inplace=True)
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


def basic_preprocess_sessions(df, mode, nrows, recompute=False):
    """
    Trigger the whole basic processing steps
    :param df: raw dataframe
    :param mode: 'train' or 'test'
    :param nrows: number of rows that was used to the input df
    :param recompute:
    :return: basic cleaned dataframe
    """
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, f'basic_preprocessed_{mode}_{nrows}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load from existing basic pre-processing file: '{filename}'")
        df = pd.read_parquet(filename)
    else:
        # 0) drop duplicates
        df = remove_duplicates(df)
        # 1) clipping to last click-out if any
        logger.info('Clipping session dataframe up to last click out (if there is clickout)')
        clip_up_to_last_click_ = partial(clip_up_to_last_click, mode=mode)
        df = df.groupby('session_id').apply(clip_up_to_last_click_).reset_index(drop=True)
        # 2) select only valid sessions
        logger.info(f'Checking valid sessions, {mode} length: {len(df):,}')
        filter_and_check_ = partial(filter_and_check, mode=mode)

        valid_clicked = df.groupby('session_id').apply(filter_and_check_)
        # if one right, there should be no invalid sessions at this point
        invalid_session_count = (~valid_clicked).sum()
        assert invalid_session_count == 0, f'There are {invalid_session_count} invalid sessions'
        logger.info(f"Saving '{filename}'")
        df.to_parquet(filename, index=False)
    return df


def create_action_type_mapping(group=False):
    if group:
        action_type2natural = {'clickout item': 0,
                               'search for poi': 1,
                               'search for destination': 1,
                               'search for item': 2,
                               'interaction item image': 2,
                               'interaction item info': 2,
                               'interaction item deals': 2,
                               'interaction item rating': 2,
                               'filter selection': 3,
                               'change of sort order': 4}
        n_unique_actions = len(set(action_type2natural.values()))
    else:
        action_type2natural = {'clickout item': 0,
                               'search for poi': 1,
                               'search for destination': 2,
                               'search for item': 3,
                               'interaction item image': 4,
                               'interaction item info': 5,
                               'interaction item deals': 6,
                               'interaction item rating': 7,
                               'filter selection': 8,
                               'change of sort order': 9}
        n_unique_actions = len(set(action_type2natural.values()))

    return action_type2natural, n_unique_actions


def city2country(df, recompute=False):
    logger.info('Extract country info from city column and then drop city column')
    df['country'] = df['city'].str.split(', ').str[-1]
    df['country'] = df['country'].str.lower()
    df.drop('city', axis=1, inplace=True)
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, 'country_mapping.npy')

    # then mapper to int
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load country mapping from existing: '{filename}'")
        countries_mapping = np.load(filename).item()
    else:
        unique_countries = df['country'].unique()
        countries_mapping = {v: k for k, v in enumerate(unique_countries)}

    df['country'] = df['country'].map(countries_mapping)
    return df


def platform2int(df, recompute=False):
    logger.info('Converting platform to int')
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, 'platform_mapping.npy')

    # then mapper to int
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load platform_mapping from existing: '{filename}'")
        platform_mapping = np.load(filename).item()
    else:
        plats = df['platform'].unique()
        platform_mapping = {v: k for k, v in enumerate(plats)}
    df['platform'] = df['platform'].map(platform_mapping)
    return df


def device2int(df, recompute=False):
    logger.info('Converting device to int')
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, 'device_mapping.npy')

    # then mapper to int
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load device_mapping from existing: '{filename}'")
        device_mapping = np.load(filename).item()
    else:
        devices = df['device'].unique()
        device_mapping = {v: k for k, v in enumerate(devices)}
    df['device'] = df['device'].map(device_mapping)
    return df


def preprocess_data(mode, nrows=None, add_test=True, recompute=False):
    nrows_str = 'all' if nrows is None else nrows
    add_test_str = 'test_added' if add_test else 'no_test_added'
    filename = os.path.join(Filepath.gbm_cache_path, f'preprocess_{mode}_{nrows_str}_{add_test_str}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load from existing '{filename}'")
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
        # get time
        df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.datetime.utcfromtimestamp(ts))

        # basic pre-process data i.e. dropping duplicates, only take sessions with clicks and clip to last click out
        df = basic_preprocess_sessions(df, mode=mode, nrows=nrows, recompute=recompute)

        # get country, device, platform transformed
        df = city2country(df, recompute=False)
        df = device2int(df, recompute=False)
        df = platform2int(df, recompute=False)

        logger.info('Converting action_types to int (natural number)')
        action_type2natural, _ = create_action_type_mapping()
        df['action_type'] = df['action_type'].map(action_type2natural)
        df['action_type'] = df['action_type'].astype(int)

        # select columns that get used
        df = df[USE_COLS]
        logger.debug(f'\n{df.head()}')

        # logger.info('Sort df by session_id, timestamp, step')
        # df = df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
        df.to_parquet(filename, index=False)
    logger.info(f'Dataframe from preprocess_data has shape: ({df.shape[0]:,}, {df.shape[1]})')
    logger.info('='*60)
    return df


if __name__ == '__main__':
    mode = 'train'
    nrows = 1000000
    add_test = False
    preprocess_data(mode, nrows=nrows, add_test=add_test, recompute=True)
