import time
import pandas as pd
import numpy as np
import datetime
import os
import gc
from functools import partial
import multiprocessing as mp
from scipy.stats import rankdata
from utils import load_data, get_logger, get_data_path
from clean_session import preprocess_sessions


logger = get_logger('create_model_inputs')
Filepath = get_data_path()


def flogger(df, name):
    logger.info(f'{name} shape: ({df.shape[0]:,}, {df.shape[1]})')


def create_action_type_mapping(recompute=False):
    filepath = Filepath.gbm_cache_path
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


def meta_encoding(recompute=False):
    filepath = Filepath.gbm_cache_path
    filename = os.path.join(filepath, 'meta_mapping.npy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing file: {filename}')
        meta_mapping = np.load(filename).item()
        n_properties = len(meta_mapping[list(meta_mapping.keys())[0]])
    else:
        logger.info('Load meta data')
        meta_df = load_data('item_metadata')
        logger.info('Lower and split properties str to list')
        meta_df['properties'] = meta_df['properties'].str.lower().str.split('|')
        logger.info('Get all unique properties')
        unique_properties = list(set(np.concatenate(meta_df['properties'].values)))
        property2natural = {v: k for k, v in enumerate(unique_properties)}
        n_properties = len(unique_properties)
        logger.info(f'Total number of unique meta properties: {n_properties}')
        logger.info('Convert the properties to ohe and superpose for each item_id')
        meta_df['properties'] = meta_df['properties'].apply(lambda ps: [property2natural[p] for p in ps])
        meta_df['properties'] = meta_df['properties'].apply(lambda ps: np.sum(np.eye(n_properties, dtype=np.int16)[ps],
                                                                              axis=0))
        logger.info('Create mappings')
        meta_mapping = dict(meta_df[['item_id', 'properties']].values)
        # add a mapping (all zeros) for the padded impression ids
        # meta_mapping[0] = np.zeros(n_properties, dtype=int)
        logger.info('Saving meta mapping')
        np.save(filename, meta_mapping)
    return meta_mapping, n_properties


def create_cfs_mapping(select_n_filters=32, recompute=False):
    filename = os.path.join(Filepath.gbm_cache_path, 'filters_mapping.npy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load filters mapping from existing: {filename}')
        filters2natural = np.load(filename).item()
    else:
        train = load_data('train', usecols=['current_filters'])
        test = load_data('test', usecols=['current_filters'])
        tt = pd.concat([train, test], axis=0, ignore_index=True)
        del train, test
        gc.collect()
        tt['current_filters'] = tt['current_filters'].str.lower()
        tt['current_filters'] = tt['current_filters'].str.split('|')
        tt.dropna(subset=['current_filters'], inplace=True)
        cfs = np.concatenate(tt['current_filters'].values)
        cfs_ctn = pd.value_counts(cfs, normalize=True) * 100
        # choose the top 32
        selected_filters = cfs_ctn.index[:select_n_filters].values
        rest_filters = cfs_ctn.index[select_n_filters:].values
        logger.info(f'select filters:\n{selected_filters} which covers {cfs_ctn.cumsum().iloc[select_n_filters-1]:.2f}%'
                    ' of all not nan current_filters')

        filters2natural = {v: k for k, v in enumerate(selected_filters)}
        # fit the rest to one mapping
        for f in rest_filters:
            filters2natural[f] = len(selected_filters)
        np.save(filename, filters2natural)
    return filters2natural, select_n_filters+1


def prepare_data(mode, nrows=None, convert_action_type=True, add_test=True, recompute=False):
    nrows_str = 'all' if nrows is None else nrows
    add_test_str = 'with_test' if add_test else 'no_test'
    filename = os.path.join(Filepath.gbm_cache_path, f'{mode}_{nrows_str}_{add_test_str}.snappy')

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
        df = preprocess_sessions(df, mode=mode, nrows=nrows, drop_duplicates=True, save=True, recompute=recompute)
        if mode == 'test':
            # then load the test that we need to submit
            test_sub = load_data('submission_popular')
            sub_sids = test_sub['session_id'].unique()
            df = df[df['session_id'].isin(sub_sids)].reset_index(drop=True)
            flogger(df, 'Load test with only what ids needed for submissions')

        # get time and select columns that get used
        df['timestamp'] = df['timestamp'].apply(lambda ts: datetime.datetime.utcfromtimestamp(ts))
        # use_cols = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters',
        #             'reference', 'impressions', 'prices']
        use_cols = ['session_id', 'timestamp', 'step', 'action_type', 'current_filters',
                    'reference', 'impressions', 'prices']
        df = df[use_cols]
        if convert_action_type:
            logger.info('Converting action_types to int (natural number)')
            action_type2natural, _ = create_action_type_mapping(recompute=False)
            df['action_type'] = df['action_type'].map(action_type2natural)
        logger.info('Sort df by session_id, timestamp, step')
        df = df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
        df.drop('step', axis=1, inplace=True)
        flogger(df, f'Prepared {mode} data')
        df.to_parquet(filename)
    return df


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

    # in case there is just one row, use list comprehension instead of np concat
    impressions = df['impressions'].dropna().values
    unique_items = list(set([j for i in impressions for j in i] + list(df['reference'].unique())))
    temp_mapping = {v: k for k, v in enumerate(unique_items)}
    df['reference_natural'] = df['reference'].map(temp_mapping)
    click_out_mask = df['action_type'] == 0
    df.drop('action_type', axis=1, inplace=True)
    other_mask = ~click_out_mask
    prev_click_cols = [f'prev_click{i}' for i in range(len(unique_items))]
    prev_interact_cols = [f'prev_interact{i}' for i in range(len(unique_items))]

    prev_click_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[click_out_mask]['reference_natural'].values],
                                 columns=prev_click_cols, index=df[click_out_mask].index)

    prev_interact_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[other_mask]['reference_natural'].values],
                                    columns=prev_interact_cols, index=df[other_mask].index)
    df = pd.concat([df, prev_click_df, prev_interact_df], axis=1)
    df.drop('reference_natural', axis=1, inplace=True)

    # first get all previous clicks, and others
    df_temp = df.copy()
    df_temp.fillna(0, inplace=True) # we need to fillna otherwise the cumsum would not compute on row with nan value
    prev_click_df = df_temp[prev_click_cols].shift(1).cumsum()
    prev_interact_df = df_temp[prev_interact_cols].shift(1).cumsum()
    # del df_temp
    # gc.collect()
    # fillna for grabing the last info (cannot ffill in the beginning as it will create extra sum in cumsum)
    df.fillna(method='ffill', inplace=True)
    # get last one
    last_click = df[prev_click_cols].shift(1)
    # name last ones
    last_click.columns = [f'last_click{i}' for i in range(len(unique_items))]
    last_interact = df[prev_interact_cols].shift(1)
    df.drop(prev_click_cols+prev_interact_cols, axis=1, inplace=True)
    # name last ones
    last_interact.columns = [f'last_interact{i}' for i in range(len(unique_items))]

    # concat all
    df = pd.concat([df, last_click, prev_click_df, last_interact, prev_interact_df], axis=1)
    # only need click-out rows now
    df = df[click_out_mask].reset_index(drop=True)
    df.fillna(0, inplace=True)

    # now actually match the corresponding ones
    def match(row, cols):
        impressions_natural = [temp_mapping[imp] for imp in row['impressions']]
        return row[cols].values[impressions_natural]

    iter_cols = {'last_click': last_click.columns, 'prev_click': prev_click_cols,
                 'last_interact': last_interact.columns, 'prev_interact': prev_interact_cols}
    for k, v in iter_cols.items():
        try:
            func = partial(match, cols=v)
            df[k] = df.apply(func, axis=1)
            df.drop(v, axis=1, inplace=True)
        except Exception as err:
            print('='*10, err)
            print(df.columns.values)
            raise Exception(err)
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
    # fts_df.to_parquet(os.path.join(Filepath.gbm_cache_path, f'{mode}_session_fts.snappy'))
    logger.info(f'\n{fts_df.head()}\n{fts_df.columns}')
    logger.info(f'Total time taken to generate fts: {(time.time()-t1)/60:.2f} mins')
    return fts_df


def save_cache(arr, name):
    filepath = Filepath.gbm_cache_path
    np.save(os.path.join(filepath, name), arr)


def expand(df, col):
    n_cols = len(df[col].iloc[0])
    expand_cols = [f'{col}_{i}' for i in range(n_cols)]
    df[expand_cols] = pd.DataFrame(df[col].values.tolist(), index=df.index)
    df.drop(col, axis=1, inplace=True)
    return df


def create_model_inputs(mode, nrows=100000, recompute=False):
    nrows_ = nrows if nrows is not None else 15932993
    logger.info(f"\n{'='*10} Creating {mode.upper()} model inputs with {nrows_:,} rows and recompute={recompute} {'='*10}")
    filename = os.path.join(Filepath.gbm_cache_path, f'{mode}_inputs_{nrows}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = prepare_data(mode, nrows=nrows, add_test=False, recompute=False)
        # ==============================================================================================================
        # process impressions
        df['impressions'] = df['impressions'].str.split('|')

        # get some numeric and interactions ============================================================================
        logger.info('Compute session features')
        df = compute_session_fts(df, mode)

        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])
        df['n_imps'] = df['impressions'].str.len()
        padding_mask = df['n_imps'] < 25
        # pad zeros
        cols_to_pad = ['impressions', 'last_click', 'prev_click', 'last_interact', 'prev_interact']
        for c in cols_to_pad:
            # convert list of pad as array gets saved without comma in csv
            df.loc[padding_mask, c] = (df.loc[padding_mask, c]
                                       .apply(lambda x: list(np.pad(x, (0, 25 - len(x)), mode='constant')) ))
        logger.debug(f'Nans:\n{df.isna().sum()}')

        # get target
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

        if mode == 'test':
            # for testing submission, keep records of the sessions_ids and impressions (still a str)
            test_sub = df[['session_id', 'impressions']]
            test_sub.to_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'), index=False)
            del test_sub

        # Save the session_ids =========================================================================================
        logger.debug(df.columns)
        train_ids = df['session_id'].values
        save_cache(train_ids, f'{mode}_ids.npy')
        df.drop('session_id', axis=1, inplace=True)
        del train_ids
        gc.collect()

        # hist_cols = ['last_click', 'prev_click', 'last_interact', 'prev_interact']
        # history = np.concatenate([np.array(list(df[c].values))[:, :, None] for c in hist_cols], axis=-1)
        # df.drop(hist_cols, axis=1, inplace=True)
        # save_cache(history, f'{mode}_history.npy')
        # del history
        # gc.collect()

        # First get price info =========================================================================================
        logger.info('Split prices str to list and convert to int')
        # click_mask = df['action_type'] == 0
        df['prices'] = df['prices'].str.split('|')
        df['prices'] = df['prices'].apply(lambda x: [float(p) for p in x])
        logger.info('Add price rank')

        def _rank_price(prices_row):
            ranks = rankdata(prices_row, method='dense')
            return ranks / (ranks.max())

        # add price rank
        df['prices_rank'] = df['prices'].apply(_rank_price)
        logger.info('Pad 0s for prices length shorter than 25')
        padding_mask = df['prices'].str.len() < 25
        df.loc[padding_mask, 'prices'] = df.loc[padding_mask, 'prices'].apply(lambda x: np.pad(x, (0, 25 - len(x)),
                                                                                               mode='constant'))
        df.loc[padding_mask, 'prices_rank'] = (df.loc[padding_mask, 'prices_rank']
                                                 .apply(lambda x: np.pad(x, (0, 25-len(x)),
                                                                         mode='constant', constant_values=np.nan)))

        # current_filters ==============================================================================================
        logger.info('Lower case current filters and split to list')
        df['current_filters'] = df['current_filters'].str.lower().str.split('|')
        cfs2natural, n_cfs = create_cfs_mapping()
        logger.info(f'There are total {n_cfs} unique filters')

        logger.info('Apply ohe superposition of filters to each records')
        df.loc[df['current_filters'].notna(), 'current_filters'] = (df.loc[df['current_filters'].notna(),
                                                                           'current_filters']
                                                                    .apply(lambda cfs: [cfs2natural[cf] for cf in cfs]))
        del cfs2natural
        gc.collect()
        # zeros if the cfs is nan (checked with type(cfs) is list not float
        df['current_filters'] = (df['current_filters'].apply(lambda cfs: np.sum(np.eye(n_cfs, dtype=np.int16)[cfs],
                                                                                axis=0)
                                                             if type(cfs) == list
                                                             else np.zeros(n_cfs, dtype=np.int16)))

        # drop not needed columns
        drop_cols = ['impressions', 'reference']
        df.drop(drop_cols, axis=1, inplace=True)
        # finally expand all column of list to columns
        expand_cols = ['prices', 'prices_rank', 'last_click', 'prev_click', 'last_interact',
                       'prev_interact', 'current_filters']
        for col in expand_cols:
            logger.info(f'Expanding on {col}')
            df = expand(df, col)

        df.to_parquet(filename)

        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')
    return df


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 100000,
            'recompute': True}
    logger.info(f'Creating data input: {args}')
    _ = create_model_inputs(**args)
