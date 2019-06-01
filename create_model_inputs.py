import time
import os
import gc
from functools import partial
import multiprocessing as mp
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from utils import load_data, get_logger, get_data_path
from preprocessing import preprocess_data

logger = get_logger('create_model_inputs')
Filepath = get_data_path()


def meta_encoding(recompute=False):
    """
    Get encoding, i.e. the properties from meta csv
    :param recompute:
    :return:
        meta_mapping: dict, mapping {123: [1, 0, 0, 1, 0, ...], ...}
        n_properties: number of unique properties
    """
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
    """
    Provide natural number representation (i.e. starting from 0 int) of top selected filters
    :param select_n_filters: number of top filters to use
    :param recompute:
    :return:
        filters2natural: dict, mapping dict e.g. {'focus on distance': 42}
        select_n_filters+1: int, plus one as the rest filters all get assigned to one number
    """
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


def compute_session_func(grp):
    """
    Main working function to compute features or shaping data
    :param grp: dataframe associated with a group key (session_id) from grouby
    :return: dataframe with feature columns
    """
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
    click_out_mask = df['action_type'] == 'clickout item'
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
            print(df.apply(func, axis=1).head())
            print(df.reference.head())
            print(df[v].head())

            raise Exception(err)
    return df


def compute_session(args):
    """
    Get assigned group ids (list of session ids) from multiprocessing and compute features using worker function with
    groupby
    :param args: tuple,
        gids: list of session_ids to work on
        df: the input dataframe (note: multiprocessing cannot share memory thus it makes copy of this df, which could
            result in memory issue)
    :return: dataframe: features, then later to be concatenated with results from other pools
    """
    # grab the args
    gids, df = args
    # selecting the assigned session ids and grouping on session level
    grps = (df[df['session_id'].isin(gids)]
            .reset_index(drop=True)
            .groupby('session_id'))

    # use apply to compute session level features
    features = grps.apply(compute_session_func).reset_index(drop=True)
    return features


def compute_session_fts(df, nprocs=None):
    """
    Initialize feature calculation with multiprocessing
    :param df: input preprocessed dataframe
    :param nprocs: numer of processes to use, if None all cores minus one is used
    :return: dataframe: input dataframe to model
    """
    t1 = time.time()
    if nprocs is None:
        nprocs = mp.cpu_count() - 1
        # nprocs = 1
        logger.info('Using {} cores'.format(nprocs))

    # get all session ids
    sids = df['session_id'].unique()
    fts = []

    # create iterator to pass in args for pool
    def args_gen():
        for i in range(nprocs):
            yield (sids[range(i, len(sids), nprocs)], df)

    # init multiprocessing pool
    pool = mp.Pool(nprocs)
    for ft in pool.map(compute_session, args_gen()):
        fts.append(ft)
    pool.close()
    pool.join()

    # concatenate results from each pool
    fts_df = pd.concat(fts, axis=0)
    logger.info(f'\n{fts_df.head()}\n{fts_df.columns}')
    logger.info(f'Total time taken to generate fts: {(time.time()-t1)/60:.2f} mins')
    return fts_df


def save_cache(arr, name):
    """
    Save array with name
    :param arr:
    :param name:
    :return:
    """
    filepath = Filepath.gbm_cache_path
    np.save(os.path.join(filepath, name), arr)


def expand(df, col):
    """
    Expand a column of list to a list of columns and drop the original column
    :param df:
    :param col: column of df to expand on
    :return:
    """
    n_cols = len(df[col].iloc[0])
    expand_cols = [f'{col}_{i}' for i in range(n_cols)]
    df[expand_cols] = pd.DataFrame(df[col].values.tolist(), index=df.index)
    df.drop(col, axis=1, inplace=True)
    return df


def create_model_inputs(mode, nrows=100000, padding_value=0, add_test=False, recompute=False):
    nrows_ = nrows if nrows is not None else 15932993
    add_test_str = 'test_added' if add_test else 'no_test_added'
    logger.info(f"\n{'='*10} Creating {mode.upper()} model inputs with {nrows_:,} rows, "
                f"{add_test_str} and recompute={recompute} {'='*10}")
    filename = os.path.join(Filepath.gbm_cache_path, f'{mode}_inputs_{nrows}_{add_test_str}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from existing {filename}')
        df = pd.read_parquet(filename)
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = preprocess_data(mode, nrows=nrows, add_test=add_test, recompute=False)
        # ==============================================================================================================

        # process impressions
        df['impressions'] = df['impressions'].str.split('|')

        # get some numeric and interactions ============================================================================
        logger.info('Compute session features')
        df = compute_session_fts(df)
        # convert impressions to int
        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])
        df['n_imps'] = df['impressions'].str.len()
        padding_mask = df['n_imps'] < 25
        # pad zeros for length less than 25
        cols_to_pad = ['impressions', 'last_click', 'prev_click', 'last_interact', 'prev_interact']
        for c in cols_to_pad:
            # convert list of pad as array gets saved without comma in csv
            df.loc[padding_mask, c] = (df
                                       .loc[padding_mask, c]
                                       .apply(lambda x: list(np.pad(x, (0, 25 - len(x)),
                                                                    mode='constant',
                                                                    constant_values=0))))
        logger.debug(f'Nans:\n{df.isna().sum()}')

        # get target
        if mode == 'train':
            logger.info('Assign target, first convert reference id to int'
                        'if needed convert non-integer str reference value to a str number')
            non_int = df['reference'].str.contains(r'[^\d]+')
            df.loc[non_int, 'reference'] = '-1'
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
            logger.info('Remove the nan target, which comes from the ones whose reference id is not in impression list')
            nan_target = df['target'].isna()
            logger.info(f'There are {nan_target.sum()} number of nan targets')
            # drop the ones whose reference is not in the impression list
            df = df[~nan_target].reset_index(drop=True)
            df['target'] = df['target'].astype(int)
            logger.info(f"\nTarget distribution: \n{pd.value_counts(df['target']).head()}")

        elif mode == 'test':
            # for testing submission, keep records of the sessions_ids and impressions
            test_sub = df[['session_id', 'impressions']]
            test_sub.to_csv(os.path.join(Filepath.sub_path, 'test_sub.csv'), index=False)
            del test_sub
        else:
            raise ValueError('Invalid mode')

        # price info ===================================================================================================
        logger.info('Split prices str to list and convert to int')
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
        df.loc[padding_mask, 'prices'] = (df
                                          .loc[padding_mask, 'prices']
                                          .apply(lambda x: np.pad(x, (0, 25 - len(x)),
                                                                  mode='constant',
                                                                  constant_values=padding_value)))
        df.loc[padding_mask, 'prices_rank'] = (df
                                               .loc[padding_mask, 'prices_rank']
                                               .apply(lambda x: np.pad(x, (0, 25-len(x)),
                                                                       mode='constant',
                                                                       constant_values=padding_value)))

        # current_filters ==============================================================================================
        logger.info('Lower case current filters and split to list')
        df['current_filters'] = df['current_filters'].str.lower().str.split('|')
        cfs2natural, n_cfs = create_cfs_mapping()
        logger.info(f'There are total {n_cfs} unique filters')

        logger.info('Apply ohe superposition of filters to each records')
        cf_not_na_mask = df['current_filters'].notna()
        df.loc[cf_not_na_mask, 'current_filters'] = (df
                                                     .loc[cf_not_na_mask, 'current_filters']
                                                     .apply(lambda cfs: [cfs2natural[cf] for cf in cfs]))
        del cfs2natural
        gc.collect()
        # zeros if the cfs is nan (checked with type(cfs) is list not float
        df['current_filters'] = (df['current_filters'].apply(lambda cfs: np.sum(np.eye(n_cfs, dtype=np.int16)[cfs],
                                                                                axis=0)
                                                             if type(cfs) == list
                                                             else np.zeros(n_cfs, dtype=np.int16)))

        # drop columns that not needed
        drop_cols = ['impressions', 'reference']
        df.drop(drop_cols, axis=1, inplace=True)

        # finally expand all column of list to columns
        expand_cols = ['prices', 'prices_rank', 'last_click', 'prev_click', 'last_interact',
                       'prev_interact', 'current_filters']
        for col in expand_cols:
            logger.info(f'Expanding on {col}')
            df = expand(df, col)

        # if test, only need to keep last row
        if mode == 'test':
            df = df.groupby('session_id').last().reset_index(drop=True)
        # save
        df.to_parquet(filename)
        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')
    logger.warning("Note: no features were normalized!")
    return df


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 100000,
            'add_test': False,
            'padding_value': np.nan,
            'recompute': True}

    logger.info(f'Creating model input: {args}')
    _ = create_model_inputs(**args)
