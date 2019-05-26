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


def prepare_data(mode, nrows=None, convert_action_type=True, add_test=True, recompute=False):
    nrows_str = 'all' if nrows is None else nrows
    add_test_str = 'with_test' if add_test else 'no_test'
    filename = os.path.join(Filepath.cache_path, f'{mode}_{nrows_str}_{add_test_str}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
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
        # pre-process data i.e. dropping duplicates, only take sessions with clicks and clip to last click out
        df = preprocess_sessions(df, mode=mode, drop_duplicates=True, save=True, recompute=recompute)
        if mode == 'test':
            # then load the test that we need to submit
            test_sub = load_data('submission_popular')
            sub_sids = test_sub['session_id'].unique()
            df = df[df['session_id'].isin(sub_sids)].reset_index(drop=True)
            flogger(df, 'Load test with only what ids needed for submissions')

        # get time and select columns that get used
        df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.datetime.utcfromtimestamp(ts))
        use_cols = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters',
                    'reference', 'impressions', 'prices']
        df = df[use_cols]
        if convert_action_type:
            logger.info('Converting action_types to int (natural number)')
            action_type2natural, _ = create_action_type_mapping(recompute=False)
            df['action_type'] = df['action_type'].map(action_type2natural)
        logger.info('Sort df by session_id, timestamp, step')
        df = df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
        flogger(df, f'Prepared {mode} data')
        df.to_parquet(filename)
    return df


def create_cfs_mapping(recompute=False):
    filename = os.path.join(Filepath.cache_path, 'filters_mapping.npy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load filters mapping from existing: {filename}')
        filters2natural = np.load(filename).item()
    else:
        train = load_data('train', usecols=['current_filters'])
        test = load_data('test', usecols=['current_filters'])
        tt = pd.concat([train, test], axis=0, ignore_index=True)
        print(tt.columns, '!' * 30)
        del train, test
        gc.collect()
        tt['current_filters'] = tt['current_filters'].str.lower()
        tt['current_filters'] = tt['current_filters'].str.split('|')
        tt.dropna(subset=['current_filters'], inplace=True)
        cfs = np.concatenate(tt['current_filters'].values)
        cfs_ctn = pd.value_counts(cfs, normalize=True) * 100
        # choose the top 32
        selected_filters = cfs_ctn.index[:32].values
        rest_filters = cfs_ctn.index[32:].values
        logger.info(f'select filters:\n{selected_filters} which covers {cfs_ctn.cumsum().iloc[31]:.2f}% '
                    'of all not nan current_filters')

        filters2natural = {v: k for k, v in enumerate(selected_filters)}
        # fit the rest to one mapping
        for f in rest_filters:
            filters2natural[f] = len(selected_filters)
        np.save(filename, filters2natural)
    nf = len(filters2natural)
    return filters2natural, nf


def compute_session_func(grp):
    df = grp.copy()
    # number of records in session
    df['session_size'] = list(range(1, len(df)+1))

    # session_time duration (subtract the min)
    df['session_duration'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # get successive time difference
    df['last_duration'] = df['timestamp'].diff().dt.total_seconds()
    df['last_duration'] = df['last_duration'].fillna(0)
    df.drop('timestamp', axis=1, inplace=True)

    # last reference id in current impression location and its action_type
    df[['ref_shift', 'at_shift']] = df[['reference', 'action_type']].shift(1)

    # previous click-outs
    # now we only need to select rows with action_type that is clickout item
    df = df[df['action_type'] == 'clickout item'].reset_index(drop=True)
    df.drop('action_type', axis=1, inplace=True)

    impressions = df['impressions'].dropna().values
    # in case there is just one row, use list comprehension instead of np concat
    unique_items = list(set([j for i in impressions for j in i] + list(df['reference'].unique())))
    temp_mapping = {v: k for k, v in enumerate(unique_items)}

    # 
    df['reference_natural'] = df['reference'].map(temp_mapping)
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


def compute_session_fts(df, mode):
    # last_rid = partial(last_reference_id, mode=mode)
    aggs = {'timestamp': [session_duration, dwell_time_prior_clickout],
            'current_filters': [last_filters],
            'session_id': 'size'}

    session_grp = df[['session_id', 'timestamp', 'current_filters']].groupby('session_id')
    session_fts = session_grp.agg(aggs)
    session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
    logger.info(f'Session features generated: {list(session_fts.columns)}')
    session_fts.reset_index(inplace=True)

    # add last_reference_id and its action_type
    last_rid = partial(last_reference_id, mode=mode)
    session_grp = df[['session_id', 'action_type', 'reference']].groupby('session_id')
    action_id_pair = session_grp.apply(last_rid).reset_index(name='action_id_pair')

    df = pd.merge(df, action_id_pair, on='session_id', how='left')
    return pd.merge(df, session_fts, on='session_id')


def save_cache(arr, name):
    filepath = Filepath.cache_path
    np.save(os.path.join(filepath, name), arr)


def create_model_inputs(mode, nrows=100000, recompute=False):
    nrows_ = nrows if nrows is not None else 15932993
    logger.info(f"\n{'='*10} Creating {mode.upper()} model inputs with {nrows_:,} rows and recompute={recompute} {'='*10}")
    filepath = Filepath.cache_path
    # filenames = ['numerics', 'impressions', 'prices', 'cfilters']
    filenames = ['numerics', 'prices', 'cfilters']

    if mode == 'train':
        filenames.append('targets')
    filepaths = [os.path.join(filepath, f'{mode}_{fn}.npy') for fn in filenames]

    if sum([os.path.isfile(f) for f in filepaths]) == len(filepaths) and not recompute:
        logger.info(f'Load from existing {filepaths}')
        model_inputs = {filenames[k]: np.load(v) for k, v in enumerate(filepaths)}
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = prepare_data(mode, nrows=nrows, add_test=False, recompute=False)
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

        logger.info('Lower case current filters and split to list')
        df['cfs'] = df['current_filters_last_filters'].str.lower().str.split('|')

        logger.info('Split prices str to list and convert to int')
        df['prices'] = df['prices'].str.split('|')
        df['prices'] = df['prices'].apply(lambda x: [int(p) for p in x])
        logger.info('Pad 0s for prices length shorter than 25')
        df['time_steps'] = df['prices'].str.len()
        padding_mask = df['time_steps'] < 25
        df.loc[padding_mask, 'prices'] = df.loc[padding_mask, 'prices'].apply(lambda x: np.pad(x, (0, 25-len(x)),
                                                                                               mode='constant'))
        # logger.info('Log1p-transform prices')
        # df['prices'] = df['prices'].apply(lambda p: np.log1p(p))
        logger.info('Normalizing price')

        # maybe normalize to percentage within each records, check does each item_id have the same price
        # over all records
        def normalize(ps):
            p_arr = np.array(ps)
            return p_arr / (p_arr.max())
        df['prices'] = df['prices'].apply(normalize)

        logger.info('Split impression str to list of impressions')
        df['impressions'] = df['impressions'].str.split('|')
        df['nimp'] = df['impressions'].str.len()/25

        logger.info('Convert impression str to int')
        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])
        # logger.info('Pad 0s for impressions length shorter than 25')
        # df.loc[padding_mask, 'impressions'] = (df.loc[padding_mask, 'impressions']
        #                                          .apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant')))

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
                return np.zeros(3, dtype=int)
            # ref = row['reference_last_reference_id']
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
                    if action_type == 'clickout item':
                        return np.array([pos, 1, 0])
                    else:
                        return np.array([pos, 0, 1])
                else:
                    return np.zeros(3, dtype=int)

        logger.info('Divide last_ref_id by 25')
        assign_last_ref_id_func = partial(assign_last_ref_id, divide=True)
        df['last_ref_ind'] = df.apply(assign_last_ref_id_func, axis=1)
        
        # # create meta ohe
        # logger.info('Load meta data')
        # meta_df = load_data('item_metadata')
        # logger.info('Lower and split properties str to list')
        # meta_df['properties'] = meta_df['properties'].str.lower().str.split('|')
        # logger.info('Get all unique properties')
        # unique_properties = list(set(np.concatenate(meta_df['properties'].values)))
        # property2natural = {v: k for k, v in enumerate(unique_properties)}
        # n_properties = len(unique_properties)
        # logger.info(f'Total number of unique meta properties: {n_properties}')
        # logger.info('Convert the properties to ohe and superpose for each item_id')
        # meta_df['properties'] = meta_df['properties'].apply(lambda ps: [property2natural[p] for p in ps])
        # meta_df['properties'] = meta_df['properties'].apply(lambda ps: np.sum(np.eye(n_properties, dtype=int)[ps],
        #                                                                       axis=0))
        # logger.info('Create mappings')
        # meta_mapping = dict(meta_df[['item_id', 'properties']].values)
        # logger.info('Saving meta mapping')
        # np.save(os.path.join(filepath, 'meta_mapping.npy'), meta_mapping)
        # # add a mapping (all zeros) for the padded impression ids
        # meta_mapping[0] = np.zeros(n_properties, dtype=int)
        # del meta_df, unique_properties, property2natural
        # gc.collect()
        #
        # logger.info('Apply meta ohe mapping to impressions (this could take some time)')
        # df['impressions'] = (df['impressions'].apply(lambda imps: np.vstack([meta_mapping[i]
        #                                              if i in meta_mapping.keys()
        #                                              else np.zeros(n_properties, dtype=int)
        #                                              for i in imps])))
        # del meta_mapping
        # gc.collect()

        logger.info('Create ohe superposition of filters')
        cfs2natural, n_cfs = create_cfs_mapping()
        logger.info(f'There are total {n_cfs} unique filters')

        logger.info('Apply ohe superposition of filters to each records')
        df.loc[df['cfs'].notna(), 'cfs'] = (df.loc[df['cfs'].notna(), 'cfs']
                                              .apply(lambda cfs: [cfs2natural[cf] for cf in cfs]))
        del cfs2natural
        gc.collect()
        # zeros if the cfs is nan (checked with type(cfs) is list not float)
        df['cfs'] = (df['cfs'].apply(lambda cfs: np.sum(np.eye(n_cfs, dtype=int)[cfs], axis=0)
                                     if type(cfs) == list else np.zeros(n_cfs, dtype=int)))

        logger.info('Grabbing list of inputs')

        # PRICES
        prices = np.array(list(df['prices'].values))
        df.drop('prices', axis=1, inplace=True)
        save_cache(prices, f'{mode}_prices.npy')

        df.drop('impressions', axis=1, inplace=True)
        # logger.info('Getting impressions')
        # # IMPRESSIONS
        # impressions = np.array(list(df['impressions'].values))
        # df.drop('impressions', axis=1, inplace=True)
        # save_cache(impressions, f'{mode}_impressions.npy')

        logger.info('Getting current_filters')
        # CURRENT_FILTERS
        cfilters = np.array(list(df['cfs'].values))
        df.drop('cfs', axis=1, inplace=True)
        save_cache(cfilters, f'{mode}_cfilters.npy')

        logger.info('Getting numerics')
        # numerics
        num_cols = ['session_id_size', 'timestamp_dwell_time_prior_clickout', 'nimp']
        logger.info('Filling nans with value=-1')
        for c in num_cols:
            # df[c] = df[c].fillna(-1)
            df[c] = df[c].fillna(0)
        numerics = np.concatenate((df[num_cols].values, np.array(list(df['last_ref_ind'].values))), axis=1)
        df.drop(num_cols, axis=1, inplace=True)
        save_cache(numerics, f'{mode}_numerics.npy')

        # model_inputs = {'numerics': numerics, 'impressions': impressions, 'prices': prices,
        #                 'cfilters': cfilters}
        model_inputs = {'numerics': numerics, 'prices': prices,
                        'cfilters': cfilters}

        if mode == 'train':
            logger.info('Getting targets')
            # TARGETS
            targets = df['target'].values
            df.drop('target', axis=1, inplace=True)
            save_cache(targets, 'train_targets.npy')
            model_inputs['targets'] = targets
        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')

    return model_inputs


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 1000000,
            'recompute': False}
    logger.info(f'Creating data input: {args}')
    _ = create_model_inputs(**args)
