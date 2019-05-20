import time
import pandas as pd
import numpy as np
import datetime
import os
import multiprocessing as mp
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


def prepare_data(mode, nrows=None, add_test=True, recompute=True):
    # first load data
    if mode == 'train':
        df = load_data(mode, nrows=nrows)
        if add_test:
            logger.info('Add available test data')
            df_test = load_data('test', nrows=nrows)
            df = pd.concat([df, df_test], axis=0, ignore_index=True)
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
    usecols = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters',
               'reference', 'impressions', 'prices']
    df = df[usecols]
    logger.info('Sort df by session_id, timestamp, step')
    df = df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
    flogger(df, f'Prepared {mode} data')
    return df


def compute_session_func(grp):
    df = grp.copy()
    # number of records in session
    df['session_size'] = list(range(1, len(df)+1))

    # session_time duration (subtract the min)
    t_init = df['timestamp'].min()
    df['session_duration'] = (df['timestamp'] - t_init).dt.total_seconds()

    # get successive time difference
    df['last_duration'] = df['timestamp'].diff().dt.total_seconds()
    # df['last_duration'] = df['last_duration'].fillna(0) # do not fillna with 0 as it would indicate this is first row
    df.drop('timestamp', axis=1, inplace=True)

    # last reference id in current impression location and its action_type
    df[['ref_shift', 'at_shift']] = df[['reference', 'action_type']].shift(1)

    # previous click-outs
    # now we only need to select rows with action_type that is clickout item
    df = df[df['action_type'] == 'clickout item'].reset_index(drop=True)
    df.drop('action_type', axis=1, inplace=True)

    impressions = df['impressions'].dropna().values
    unique_items = list(set([j for i in impressions for j in i] + list(df['reference'].unique())))

    mapping = {v: k for k, v in enumerate(unique_items)}
    df['reference_natural'] = df['reference'].map(mapping)
    prev_cols = [f'prev_{i}' for i in range(len(unique_items))]
    reference_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df['reference_natural'].values],
                                columns=prev_cols,
                                index=df.index)
    df.drop('reference_natural', axis=1, inplace=True)
    df = pd.concat([df, reference_df], axis=1)
    df[prev_cols] = df[prev_cols].cumsum().shift(1)
    # df[prev_cols] = df[prev_cols].fillna(0) # do not fillna as it would indicate it's the first row

    def match(row):
        impressions_natural = [mapping[imp] for imp in row['impressions']]
        return row[prev_cols].values[impressions_natural]
    df['prev_clickouts'] = df.apply(match, axis=1)
    # remove the prev ohe
    df.drop(prev_cols, axis=1, inplace=True)

    # come back to finding last reference relative location
    def find_relative_loc(row):
        if row['session_size'] == 1:
            return np.nan
        else:
            ref_shift = row['ref_shift']
            row_impressions = list(row['impressions'])
            if ref_shift in row_impressions:
                return row_impressions.index(ref_shift)
            else:
                return -1

    df['ref_shift'] = df.apply(find_relative_loc, axis=1)
    return df


def compute_session(args):
    # grab the args
    gids, df = args
    # selecting the assigned session ids and grouping on session level
    grps = (df[df['session_id'].isin(gids)]
            .reset_index(drop=True)
            .groupby('session_id'))

    # use apply to compute session level features
    features = grps.apply(compute_session_func).reset_index(drop=True)
    return features


def compute_session_fts(df, mode=None, nprocs=None):
    # some processing before entering groupby
    df['impressions'] = df['impressions'].str.split('|')
    df['n_imps'] = df['impressions'].str.len()
    # number of current filters
    df['n_cf'] = df['current_filters'].str.split('|').str.len()
    df['n_cf'] = df['n_cf'].fillna(0)
    df.drop('current_filters', axis=1, inplace=True)

    t1 = time.time()
    if nprocs is None:
        nprocs = mp.cpu_count() - 1
        logger.info('Using {} cores'.format(nprocs))

    sids = df['session_id'].unique()

    fts = []

    # create iterator to pass in args
    def args_gen():
        for i in range(nprocs):
            yield (sids[range(i, len(sids), nprocs)], df)
    # init multiprocessing pool
    pool = mp.Pool(nprocs)
    for ft in pool.map(compute_session, args_gen()):
        fts.append(ft)
    pool.close()
    pool.join()
    fts_df = pd.concat(fts, axis=0)
    fts_df.to_parquet(os.path.join(Filepath.cache_path, f'{mode}_session_fts.snappy'))
    logger.info(f'Total time taken to generate fts: {(time.time()-t1)/60:.2f}mins')
    return fts_df


def save_cache(arr, name):
    filepath = Filepath.cache_path
    np.save(os.path.join(filepath, name), arr)


def create_model_inputs(mode, nrows=100000, recompute=False):
    nrows_ = nrows if nrows is not None else 15932993
    logger.info(f"\n{'='*20}\nCreating {mode.upper()} model inputs with {nrows_:,} rows"
                f" and recompute={recompute}\n{'='*20}")
    filename = os.path.join(Filepath.cache_path, f'{mode}_inputs.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = prepare_data(mode, nrows=nrows, recompute=True)
        logger.info('Compute session features')
        df = compute_session_fts(df, mode)
        flogger(df, 'df shape after compute fts')

        if mode == 'test':
            logger.info('Only select last click-out from each session')
            df = df.groupby('session_id').last().reset_index()

        # log-transform on session_size feature
        logger.info('Log-transform on session_size feature')
        df['session_size'] = np.log(df['session_size'])

        # log1p-transform on timestamp_dwell_time_prior_clickout but will cliping upper to 1hr
        logger.info('Also log-transform on timestamp_dwell_time_prior_clickout but will cliping upper to 1hr')
        df['last_duration'] = np.log1p(df['last_duration'].clip(upper=60 ** 2))

        if mode == 'test':
            # for testing submission, keep records of the sessions_ids and impressions (still a str)
            test_sub = df[['session_id', 'impressions']]
            test_sub.to_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'), index=False)
            del test_sub

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

        # maybe normalize to percentage within each records
        logger.info('Normalizing price')

        def normalize(ps):
            p_arr = np.array(ps)
            return p_arr / np.nanmax(p_arr)

        df['prices_percentage'] = df['prices'].apply(normalize)
        df[[f'price_{i}' for i in range(25)]] = pd.DataFrame(df['prices_percentage'].values.tolist(), index=df.index)
        df.drop(['prices', 'prices_percentage'], axis=1, inplace=True)

        # convert impressions and reference to int
        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])
        logger.info('Pad 0s for impressions length shorter than 25')
        df.loc[padding_mask, 'impressions'] = (df.loc[padding_mask, 'impressions']
                                                 .apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant')))

        # pad the prev_cos
        logger.info('Pad previous click-out one-hot indicator')
        df.loc[padding_mask, 'prev_clickouts'] = (df.loc[padding_mask, 'prev_clickouts']
                                                  .apply(lambda x: np.pad(x, (0, 25 - len(x)), mode='constant',
                                                                          constant_values=np.nan)))
        df[[f'prev_clickouts{i}' for i in range(25)]] = pd.DataFrame(df['prev_clickouts'].values.tolist(),
                                                                     index=df.index)
        df.drop(['prev_clickouts'], axis=1, inplace=True)

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

        # convert at(action_atype to int)
        at_mapping, _ = create_action_type_mapping()
        df['at_shift'] = df['at_shift'].map(at_mapping)

        logger.debug('Saving session_ids for verification purposes')
        np.save(os.path.join(Filepath.cache_path, f'{mode}_session_ids.npy'), df['session_id'].values)

        drop_cols = ['session_id', 'impressions', 'reference']

        logger.info(f'Drop columns: {drop_cols}')
        df.drop(drop_cols, axis=1, inplace=True)
        logger.info(f'Generated {mode}_inputs columns:\n{df.columns}')
        logger.info(f'Number of nans in each columns:\n{df.isna().sum()}')
        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')
        df.to_parquet(filename)
    return df


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 5000000,
            'recompute': True}
    logger.info(f'Creating data input: {args}')
    _ = create_model_inputs(**args)
