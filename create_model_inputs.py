import time
import pandas as pd
import numpy as np
import datetime
import os
import gc
from functools import partial
from utils import load_data, get_logger, get_data_path
from clean_session import preprocess_sessions


logger = get_logger('create_model_inputs')
Filepath = get_data_path()


def flogger(df, name):
    logger.info(f'{name} shape: ({df.shape[0]:,}, {df.shape[1]})')


def create_action_type_mapping(recompute=False):
    filepath = Filepath.cache_path
    filename = os.path.join(filepath, 'action_types_mapping.npy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load action_types mapping from existing: {filename}')
        action_type2natural = np.load(filename).item()
        n_unique_actions = len(action_type2natural)
    else:
        # hardcode
        actions = ['clickout item', 'search for poi', 'interaction item image',
                   'interaction item info', 'interaction item deals',
                   'search for destination', 'filter selection',
                   'interaction item rating', 'search for item',
                   'change of sort order']
        action_type2natural = {v: k for k, v in enumerate(actions)}
        n_unique_actions = len(actions)
        np.save(filename, action_type2natural)
    return action_type2natural, n_unique_actions


def prepare_data(mode, convert_action_type=True, nrows=None, recompute=True):
    # first load data
    if mode == 'train':
        df = load_data(mode, nrows=nrows)
    else:
        df = load_data(mode)
    flogger(df, f'raw {mode}')
    # preprocess data i.e. dropping duplicates, only take sessions with clicks and clip to last click out
    df = preprocess_sessions(df, mode=mode, drop_duplicates=True, save=True, recompute=recompute)
    if mode == 'test':
        # then load the test that we need to submit
        test_sub = load_data('submission_popular')
        sub_sids = test_sub['session_id'].unique()
        df = df[df['session_id'].isin(sub_sids)].reset_index(drop=True)
        flogger(df, 'Load test with only what ids needed for submissions')

    # get time and select columns that get used
    df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.datetime.utcfromtimestamp(ts))
    usecols = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'current_filters',
               'reference', 'impressions', 'prices']
    df = df[usecols]
    if convert_action_type:
        logger.info('Converting action_types to int (natural number)')
        action_type2natural, _ = create_action_type_mapping(recompute=False)
        df['action_type'] = df['action_type'].map(action_type2natural)
    logger.info('Sort df by user_id, session_id, timestamp, step')
    df = df.sort_values(by=['user_id', 'session_id', 'timestamp', 'step']).reset_index(drop=True)
    flogger(df, f'Prepared {mode} data')
    return df


# create some session features
def session_duration(ts):
    if len(ts) == 1:
        return np.nan
    else:
        return (ts.max() - ts.min()).total_seconds()


def dwell_time_prior_clickout(ts):
    if len(ts) == 1:
        return np.nan
    else:
        ts_sorted = ts.sort_values()
        return (ts_sorted.iloc[-1] - ts_sorted.iloc[-2]).total_seconds()


def last_reference_id(grp, mode):
    mask = grp['reference'].notna()
    if mode == 'train':
        n = 1
        last_ref_loc = -2
    elif mode == 'test':
        n = 0
        last_ref_loc = -1
    else:
        raise ValueError(f'Invalid mode: {mode}')
    if mask.sum() <= n:
        return np.nan
    else:
        # the second last reference id i.e. the one before click out and the associated action_type
        return grp[mask]['action_type'].iloc[last_ref_loc], grp[mask]['reference'].iloc[last_ref_loc]


def previous_clickouts(grp):
    nimps = grp.iloc[-1]['nimps']
    if len(grp) == 1:
        return [0.]*int(nimps)
    else:
        prev_grp = grp.iloc[:-1]
        prev_grp_clickouts = prev_grp[prev_grp['action_type'] == 'clickout item']['reference'].unique()
        last_imps = grp.iloc[-1]['impressions'].split('|')
        return [1. if imp in prev_grp_clickouts else 0. for imp in last_imps]


def compute_session_fts(df, mode, recompute=False):
    filename = os.path.join(Filepath.cache_path, f'{mode}_session_fts.snappy')
    if os.path.isfile(filename) and not recompute:
        # so far this should not be used as in prev_cos has list
        df = pd.read_parquet(filename)
    else:
        # last_rid = partial(last_reference_id, mode=mode)
        aggs = {'timestamp': [session_duration, dwell_time_prior_clickout],
                # 'current_filters': [last_filters],
                'session_id': 'size'}

        session_grp = df[['session_id', 'timestamp']].groupby('session_id')
        session_fts = session_grp.agg(aggs)
        session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
        logger.info(f'Session features generated: {list(session_fts.columns)}')
        session_fts.reset_index(inplace=True)
        df = pd.merge(df, session_fts, on='session_id')

        # add last_reference_id and its action_type
        logger.info('Add last_reference_id and its action_type')
        last_rid = partial(last_reference_id, mode=mode)
        session_grp = df[['session_id', 'action_type', 'reference']].groupby('session_id')
        action_id_pair = session_grp.apply(last_rid).reset_index(name='action_id_pair')
        df = pd.merge(df, action_id_pair, on='session_id', how='left')

        # add previous click-out info
        logger.info('Add previous click-out info')
        df['nimps'] = df['impressions'].str.split('|').str.len()
        session_grp = df[['session_id', 'action_type', 'reference', 'impressions', 'nimps']].groupby('session_id')
        prev_cos = session_grp.apply(previous_clickouts).reset_index(name='prev_cos')
        prev_cos.to_csv('test_prev_cos.csv', index=False)
        df = pd.merge(df, prev_cos, on='session_id', how='left')
        df.drop('nimps', axis=1, inplace=True)

    return df


def save_cache(arr, name):
    filepath = Filepath.cache_path
    np.save(os.path.join(filepath, name), arr)


def create_model_inputs(mode, nrows=100000, recompute=False):
    nrows_ = nrows if nrows is not None else 15932993
    logger.info(f"\n{'='*10} Creating {mode.upper()} model inputs with {nrows_:,} rows and recompute={recompute} {'='*10}")
    filename = os.path.join(Filepath.cache_path, f'{mode}_inputs.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = prepare_data(mode, convert_action_type=False, nrows=nrows, recompute=True)
        logger.info('Compute session features')
        df = compute_session_fts(df, mode)

        logger.info('Only select last click-out from each session')
        df = df.groupby('session_id').last().reset_index()
        flogger(df, 'df shape after only selecting last click-out row each session')

        # log-transform on session_size feature
        logger.info('Log-transform on session_size feature')
        df['session_id_size'] = np.log(df['session_id_size'])

        # log1p-transform on timestamp_dwell_time_prior_clickout but will cliping upper to 1hr
        logger.info('Also log-transform on timestamp_dwell_time_prior_clickout but will cliping upper to 1hr')
        df['timestamp_dwell_time_prior_clickout'] = np.log1p(df['timestamp_dwell_time_prior_clickout'].clip(upper=60 ** 2))

        if mode == 'test':
            # for testing submission, keep records of the sessions_ids and impressions (still a str)
            test_sub = df[['session_id', 'impressions']]
            test_sub.to_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'), index=False)
            del test_sub

        # number of current filters
        df['nf'] = df['current_filters'].str.split('|').str.len()
        df.drop('current_filters', axis=1, inplace=True)

        logger.info('Split prices str to list and convert to int')
        df['prices'] = df['prices'].str.split('|')
        df['prices'] = df['prices'].apply(lambda x: [float(p) for p in x])
        logger.info('Pad 0s for prices length shorter than 25')
        df['time_steps'] = df['prices'].str.len()
        padding_mask = df['time_steps'] < 25
        df.drop('time_steps', axis=1, inplace=True)
        df.loc[padding_mask, 'prices'] = df.loc[padding_mask, 'prices'].apply(lambda x: np.pad(x, (0, 25-len(x)),
                                                                                               mode='constant',
                                                                                               constant_values=np.nan))
        # logger.info('Log1p-transform prices')
        # df['prices'] = df['prices'].apply(lambda p: np.log1p(p))
        logger.info('Normalizing price')

        # maybe normalize to percentage within each records, check does each item_id have the same price
        # over all records
        def normalize(ps):
            p_arr = np.array(ps)
            return p_arr / (p_arr.max())
        df['prices_percentage'] = df['prices'].apply(normalize)
        df[[f'price_{i}' for i in range(25)]] = pd.DataFrame(df['prices_percentage'].values.tolist(), index=df.index)
        df.drop(['prices', 'prices_percentage'], axis=1, inplace=True)

        logger.info('Split impression str to list of impressions')
        df['impressions'] = df['impressions'].str.split('|')
        df['n_imps'] = df['impressions'].str.len()
        logger.info('Convert impression str to int')
        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])
        logger.info('Pad 0s for impressions length shorter than 25')
        df.loc[padding_mask, 'impressions'] = (df.loc[padding_mask, 'impressions']
                                                 .apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant')))

        # pad the prev_cos
        logger.info('Pad previous click-out one-hot indicator')
        # print(f'!!!!!!!!!!!\n{df.prev_cos.head()}\n{df[df.prev_cos.isna()].head()}')
        df.loc[padding_mask, 'prev_cos'] = (df.loc[padding_mask, 'prev_cos']
                                              .apply(lambda x: np.pad(x, (0, 25 - len(x)), mode='constant',
                                                                      constant_values=np.nan)))
        df[[f'prev_cos_{i}' for i in range(25)]] = pd.DataFrame(df['prev_cos'].values.tolist(), index=df.index)
        df.drop(['prev_cos'], axis=1, inplace=True)

        if mode == 'train':
            logger.info('Assign target')
            logger.info('Convert reference id to int')
            df['reference'] = df['reference'].astype(int)

            # filter out nan rows with reference_id not in impressions list, since if the true target in test
            # is not in the impression list then it would not get evaluated
            def assign_target(row):
                ref = row['reference']
                imp = list(row['impressions'])
                if ref in imp:
                    return imp.index(ref)
                else:
                    return np.nan

            df['target'] = df.apply(assign_target, axis=1)
            logger.info('Remove the ones whose reference id is not in impression list')
            # drop the ones whose reference is not in the impression list
            df = df[df['target'].notna()].reset_index(drop=True)
            df['target'] = df['target'].astype(int)
            logger.info(f"Target distribution: \n{pd.value_counts(df['target']).head()}")

        logger.info('Assign location of previous reference id')

        def assign_last_ref_id(row, divide=True):
            action_id_pair = row['action_id_pair']
            if pd.isna(action_id_pair):
                # return np.zeros(2, dtype=int)
                return [np.nan]*2

            # although reference_id got converted to int, but the reference_last_reference_id was calculated
            # when it was still str value, so here we look up the index in str of impressions
            else:
                imp = [str(i) for i in row['impressions']]
                action_type, ref = action_id_pair
                if ref in imp:
                    if divide:
                        pos = (imp.index(ref) + 1) / len(imp)
                    else:
                        pos = imp.index(ref) + 1
                    # if action_type == 2:
                    #     return [pos, 1]
                    # else:
                    #     return [pos, 0]
                    return [pos, action_type]
                else:
                    return [np.nan, action_type]

        logger.info('Divide last_ref_id by 25')
        assign_last_ref_id_func = partial(assign_last_ref_id, divide=True)
        df['last_ref_ind'] = df.apply(assign_last_ref_id_func, axis=1)
        df[['pos', 'at']] = pd.DataFrame(df['last_ref_ind'].values.tolist(), index=df.index)

        logger.debug('Saving session_ids for verification purposes')
        np.save(os.path.join(Filepath.cache_path, f'{mode}_session_ids.npy'), df['session_id'].values)

        drop_cols = ['session_id', 'user_id', 'impressions', 'timestamp', 'action_type',
                     'reference', 'action_id_pair',
                     'last_ref_ind']

        logger.info(f'Drop columns: {drop_cols}')
        df.drop(drop_cols, axis=1, inplace=True)
        logger.info(f'Generated {mode}_inputs columns:\n{df.columns}')
        logger.info(f'Number of nans in each columns:\n{df.isna().sum()}')
        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')
        df.to_parquet(filename)
    return df


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 1000000,
            'recompute': True}
    logger.info(f'Creating data input: {args}')
    _ = create_model_inputs(**args)
