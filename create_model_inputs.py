import time
import os
import gc
from functools import partial
import multiprocessing as mp
from scipy.stats import rankdata
import pandas as pd
import numpy as np
from utils import load_data, get_logger, get_data_path
from preprocessing import preprocess_data, create_action_type_mapping

logger = get_logger('create_model_inputs')
Filepath = get_data_path()

CATEGORICAL_COLUMNS = ['country', 'device', 'platform', 'fs', 'sort_order', 'last_action_type']


def meta_encoding(recompute=False):
    filepath = Filepath.cache_path
    filename = os.path.join(filepath, 'meta_encodings.csv')
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load from existing file: '{filename}'")
        encoding = pd.read_csv(filename)
    else:
        meta = load_data('item_metadata')
        # get list of properties
        meta['properties'] = meta['properties'].str.lower().str.split('|')

        # create mapping
        properties = list(set(np.concatenate(meta['properties'].values)))
        property_mapping = {v: k for k, v in enumerate(properties)}
        property_names = list(property_mapping.keys())
        meta['properties'] = meta['properties'].apply(lambda l: [property_mapping[p] for p in l])
        # create zeros for encoding first
        zeros = np.zeros((len(meta), len(property_mapping.keys())), dtype=int)
        # then assign
        ps = meta['properties']
        for i in range(meta.shape[0]):
            zeros[i, ps[i]] = 1

        encoding = pd.DataFrame(zeros, columns=property_names, index=meta.item_id).reset_index()
        encoding['item_id'] = encoding['item_id'].astype(int)
        # save
        encoding.to_csv(filename, index=False)
    return encoding


def create_cfs_mapping(select_n_filters=32, recompute=False):
    """
    Provide natural number representation (i.e. starting from 0 int) of top selected filters
    :param select_n_filters: number of top filters to use
    :param recompute:
    :return:
        filters2natural: dict, mapping dict e.g. {'focus on distance': 42}
        select_n_filters+1: int, plus one as the rest filters all get assigned to one number
    """
    filename = os.path.join(Filepath.cache_path, 'filters_mapping.npy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load current filters mapping from existing: '{filename}'")
        filters2natural = np.load(filename).item()
    else:
        train = load_data('train', usecols=['current_filters'])
        test = load_data('test', usecols=['current_filters'])
        tt = pd.concat([train, test], axis=0, ignore_index=True)
        del train, test
        gc.collect()
        tt = tt.dropna(subset=['current_filters']).reset_index(drop=True)
        tt['current_filters'] = tt['current_filters'].str.lower()
        tt['current_filters'] = tt['current_filters'].str.split('|')
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


def change_sort_order_mapping():
    mapping = {'interaction sort button': 0,
               'price only': 1,
               'price and recommended': 2,
               'distance only': 3,
               'distance and recommended': 4,
               'rating and recommended': 5,
               'rating only': 6,
               'our recommendations': 7}
    return mapping


def filter_selection_mapping(select_n_filters=32, recompute=False):
    filename = os.path.join(Filepath.cache_path, 'filters_selection_mapping.npy')
    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load filters mapping from existing: '{filename}'")
        fs2natural = np.load(filename).item()
    else:
        train = load_data('train', usecols=['action_type', 'reference'])
        test = load_data('test', usecols=['action_type', 'reference'])
        tt = pd.concat([train, test], axis=0, ignore_index=True)
        del train, test
        gc.collect()
        fs_ctn = tt[tt.action_type == 'filter selection']['reference'].value_counts(normalize=True)*100
        # choose the top 32
        selected_filters = fs_ctn.index[:select_n_filters].values
        rest_filters = fs_ctn.index[select_n_filters:].values
        logger.info(
            f'select filters:\n{selected_filters} which covers {fs_ctn.cumsum().iloc[select_n_filters - 1]:.2f}%'
            ' of all filter selection action types')

        fs2natural = {v: k for k, v in enumerate(selected_filters)}
        # fit the rest to one mapping
        for f in rest_filters:
            fs2natural[f] = len(selected_filters)
        np.save(filename, fs2natural)
    return fs2natural, select_n_filters + 1


# grab mappings
action_type2natural, n_unique_actions = create_action_type_mapping()
n_unique_cs = len(change_sort_order_mapping())
_, n_unique_fs = filter_selection_mapping(select_n_filters=32)


def find_relative_ref_loc(row):
    last_ref = row['last_reference']
    if pd.isna(last_ref) or type(row['impressions']) != list:
        # first row
        return np.nan
    else:
        # try:
        imps = list(row['impressions'])
        if last_ref in imps:
            return (imps.index(last_ref)+1)/25
        else:
            return np.nan


def compute_session_func(grp):
    """
    Main working function to compute features or shaping data
    :param grp: dataframe associated with a group key (session_id) from groupby
    :return: dataframe with feature columns
    """
    df = grp.copy()
    # number of records in session
    df['session_size'] = list(range(1, len(df)+1))

    # session_time duration (subtract the min), if it is the first row, it is always nan
    if len(df) == 1:
        df['session_duration'] = np.nan
    else:
        df['session_duration'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

    # get consecutive time difference
    df['last_duration'] = df['timestamp'].diff().dt.total_seconds()
    # maybe we don't fill na with 0 as it could be implying a very short time interval
    df.drop('timestamp', axis=1, inplace=True)

    # instead of ohe, just use it as categorical input
    # shift down
    df['last_action_type'] = df['action_type'].shift(1)
    df['fs'] = df['fs'].shift(1)
    df['sort_order'] = df['sort_order'].shift(1)

    # in case there is just one row, use list comprehension instead of np concat
    impressions = df['impressions'].dropna().values
    unique_items = list(set([j for i in impressions for j in i] + list(df['reference'].unique())))
    temp_mapping = {v: k for k, v in enumerate(unique_items)}
    df['reference_natural'] = df['reference'].map(temp_mapping)
    click_out_mask = df['action_type'] == 0  # 0 is the hard-coded encoding for clickout item
    df.drop('action_type', axis=1, inplace=True)
    other_mask = ~click_out_mask
    click_cols = [f'click_{i}' for i in range(len(unique_items))]
    interact_cols = [f'interact_{i}' for i in range(len(unique_items))]

    # create click out binary encoded dataframe
    click_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[click_out_mask]['reference_natural'].values],
                            columns=click_cols, index=df[click_out_mask].index)
    # create everything else except clickout
    interact_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[other_mask]['reference_natural'].values],
                               columns=interact_cols, index=df[other_mask].index)
    df = pd.concat([df, click_df, interact_df], axis=1)
    df.drop('reference_natural', axis=1, inplace=True)

    # first get all previous clicks, and others
    df_temp = df.copy()
    df_temp.fillna(0, inplace=True)  # we need to fill na otherwise the cumsum would not compute on row with nan value
    # prev_click_df = df_temp[click_cols].shift(1).cumsum()
    # prev_interact_df = df_temp[interact_cols].shift(1).cumsum()

    # 1) multiply one hots with the step number
    prev_click_df = df_temp[click_cols].mul(df_temp['step_no_gap'], axis=0)
    prev_interact_df = df_temp[interact_cols].mul(df_temp['step_no_gap'], axis=0)
    # 2) replace zero with nan
    prev_click_df.replace(0, np.nan, inplace=True)
    prev_interact_df.replace(0, np.nan, inplace=True)
    # 3) forward fill
    prev_click_df.fillna(method='ffill', inplace=True)
    prev_interact_df.fillna(method='ffill', inplace=True)
    # 4) shift row by 1
    prev_click_df = prev_click_df.shift(1)
    prev_interact_df = prev_interact_df.shift(1)
    # 5) subtract by step
    prev_click_df = -prev_click_df.sub(df_temp['step_no_gap'], axis=0)
    prev_interact_df = -prev_interact_df.sub(df_temp['step_no_gap'], axis=0)

    # remove the original click and interact cols, swap on the last and prev clicks and interaction cols
    df.drop(click_cols + interact_cols, axis=1, inplace=True)
    # concat all
    df = pd.concat([df, prev_click_df, prev_interact_df], axis=1)

    # add impression relative location
    df['last_reference'] = df['reference'].shift(1)
    # # although, we only look at the last integer reference
    # valid_reference = df['last_reference'].str.contains(r'^\d+$')
    # # just in case its nan (it happened in test)
    # nan_valid_reference = pd.isna(valid_reference)
    # logger.info(f'There are {nan_valid_reference.sum()} nan when filtering valid reference to create last reference feature')
    # valid_reference = valid_reference[~nan_valid_reference]
    # non_valid_reference = valid_reference[~valid_reference].index
    # df.loc[non_valid_reference, 'last_reference'] = np.nan
    # df['last_reference'] = df['last_reference'].shift(1)
    # df['last_reference'].fillna(method='ffill', inplace=True)
    df.loc[click_out_mask, 'last_reference_relative_loc'] = df.loc[click_out_mask].apply(find_relative_ref_loc, axis=1)
    df.drop('last_reference', axis=1, inplace=True)

    # only need click-out rows now
    df = df[click_out_mask].reset_index(drop=True)

    # now select only the ones that is needed for each row
    def match(row, cols):
        impressions_natural = [temp_mapping[imp] for imp in row['impressions']]
        return row[cols].values[impressions_natural]

    iter_cols = {'prev_click': click_cols, 'prev_interact': interact_cols}
    for k, v in iter_cols.items():
        func = partial(match, cols=v)
        df[k] = df.apply(func, axis=1)
        df.drop(v, axis=1, inplace=True)

    return df


def compute_session(input_args):
    """
    Get assigned group ids (list of session ids) from multiprocessing and compute features using worker function with
    groupby
    :param input_args: tuple,
        gids: list of session_ids to work on
        df: the input dataframe (note: multiprocessing cannot share memory thus it makes copy of this df, which could
            result in memory issue)
    :return: dataframe: features, then later to be concatenated with results from other pools
    """
    # grab the args
    gids, df = input_args
    # selecting the assigned session ids and grouping on session level
    grps = (df[df['session_id'].isin(gids)]
            .reset_index(drop=True)
            .groupby('session_id'))
    # use apply to compute session level features
    features = grps.apply(compute_session_func)
    features = features.reset_index(drop=True)
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
    logger.info('Using {} cores'.format(nprocs))

    # get all session ids
    sids = df['session_id'].unique()
    fts = []
    # compute_session((sids, df))

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
    filepath = Filepath.cache_path
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


def different_impressions(input_df):
    logger.info('Get binary indicator whether the impression list has changed since last impression')
    grp = input_df.groupby('session_id')

    def _diff(df):
        df['imp_shift'] = df['impressions'].shift(1)
        df['imp_shift'] = df['imp_shift'].fillna(method='ffill')
        df['imp_changed'] = (df['impressions'] != df['imp_shift'])  # .astype(int)
        # the imp_changed of first impression list is always nan
        first_imp = df['impressions'].notna()
        first_imp = first_imp[first_imp].index[0]
        df.loc[first_imp, 'imp_changed'] = np.nan
        return df

    input_df = grp.apply(_diff)
    input_df.drop('imp_shift', axis=1, inplace=True)
    # chaneg it to float
    input_df['imp_changed'] = input_df['imp_changed'].astype(float)
    logger.info('Done adding impression change indicator')
    return input_df


def padding(df, pad_mask, cols_to_pad, padding_len, padding_value):
    for c in cols_to_pad:
        # convert list of pad as array gets saved without comma in csv
        df.loc[pad_mask, c] = (df
                               .loc[pad_mask, c]
                               .apply(lambda x: list(np.pad(x, (0, padding_len - len(x)),
                                                            mode='constant',
                                                            constant_values=padding_value))))
    return df


def create_model_inputs(mode, nrows=100000, padding_value=0, add_test=False, recompute=False):
    nrows = nrows if nrows is not None else 15932993
    add_test_str = 'test_added' if add_test else 'no_test_added'
    logger.info(f"\n{'='*10} Creating {mode.upper()} model inputs with {nrows:,} rows, "
                f"{add_test_str} and recompute={recompute} {'='*10}")
    filename_base = f'{mode}_inputs_{nrows}_{add_test_str}'
    filename = os.path.join(Filepath.cache_path, f'{filename_base}.snappy')

    if os.path.isfile(filename) and not recompute:
        logger.info(f"Load from existing '{filename}'")
        df = pd.read_parquet(filename)
    else:
        logger.info(f'Prepare {mode} data')
        t_init = time.time()
        df = preprocess_data(mode, nrows=nrows, add_test=add_test, recompute=False)
        # ==============================================================================================================
        # add fs (filter selection encoding) and change of sort order encoding, see mapping in preprocessing
        fs_mask = df['action_type'] == action_type2natural['filter selection']
        sort_mask = df['action_type'] == action_type2natural['change of sort order']
        fs_mapper, _ = filter_selection_mapping(select_n_filters=32)
        change_sort_order_mapper = change_sort_order_mapping()
        df.loc[fs_mask, 'fs'] = df.loc[fs_mask, 'reference'].map(fs_mapper)
        df.loc[sort_mask, 'sort_order'] = df.loc[sort_mask, 'reference'].map(change_sort_order_mapper)

        # TODO
        # add impression change indicator
        df = different_impressions(df)

        # process impressions
        df['impressions'] = df['impressions'].str.split('|')

        # get some numeric and interactions ============================================================================
        logger.info('Compute session features')
        df = compute_session_fts(df, nprocs=10)

        # convert impressions to int
        df['impressions'] = df['impressions'].apply(lambda x: [int(i) for i in x])

        # add meta encodings of ratings and stars
        logger.info('Adding ratings and stars from meta')
        meta_mapping = meta_encoding()
        rating_cols = ['satisfactory rating', 'good rating', 'very good rating', 'excellent rating']
        # create rating column in meta by summing over all the ratings
        meta_mapping['ratings'] = meta_mapping[rating_cols].sum(axis=1)
        # create star column in meta by multiplying the one hot with the star number
        star_cols = ['1 star', '2 star', '3 star', '4 star', '5 star']
        for k, v in enumerate(star_cols):
            meta_mapping[v] = meta_mapping[v] * (k+1)
        # use the max as star value
        meta_mapping['stars'] = meta_mapping[star_cols].max(axis=1)
        meta_mapping = meta_mapping[['item_id', 'ratings', 'stars']]
        # meta_items = meta_mapping['item_id'].unique()
        item_rating_mapping = dict(meta_mapping[['item_id', 'ratings']].values)
        item_star_mapping = dict(meta_mapping[['item_id', 'stars']].values)
        del meta_mapping
        gc.collect()

        # instead of iterate over each element in row, try to explode and map then collapse
        logger.info('Start rating mapping by exploding first, map then collapse')
        n_nan_imps = df['impressions'].isna().sum()
        assert n_nan_imps == 0, f'There should be 0 nan impressions but there are {n_nan_imps}'
        imp_vals = df['impressions'].tolist()
        # grab the required repeating len
        imps_lens = [len(imps) for imps in imp_vals]
        temp = pd.concat([np.repeat(df['session_id'], imps_lens), np.repeat(df['step'], imps_lens)], axis=1)

        base = pd.DataFrame(np.column_stack((temp, np.concatenate(imp_vals))),
                            columns=['session_id', 'step', 'impressions'])
        base['ratings'] = base['impressions'].map(item_rating_mapping)
        base['stars'] = base['impressions'].map(item_star_mapping)
        base.drop('impressions', axis=1, inplace=True)
        del item_rating_mapping, item_star_mapping  # , meta_items
        gc.collect()
        # collapse it back
        ratings = base.groupby(['session_id', 'step'])['ratings'].apply(list).reset_index()
        stars = base.groupby(['session_id', 'step'])['stars'].apply(list).reset_index()
        del base
        gc.collect()
        logger.info('Created ratings and stars dataframe, now merge it back to df')

        # join it back to df
        df = pd.merge(df, ratings, on=['session_id', 'step'])
        del ratings
        gc.collect()
        df = pd.merge(df, stars, on=['session_id', 'step'])
        del stars
        gc.collect()

        logger.info('Done meta mapping')
        logger.info('Add mean and median of both ratings and stars')

        def _add_mean_median_std(values):
            return pd.Series([np.nanmean(values), np.nanmedian(values), np.nanstd(values)])

        df[['mean_rating', 'median_rating', 'std_rating']] = df.apply(lambda row: _add_mean_median_std(row['ratings']),
                                                                      axis=1)
        df[['mean_star', 'median_star', 'std_star']] = df.apply(lambda row: _add_mean_median_std(row['stars']), axis=1)

        # logger.info('Transform ratings and stars to relative rank')

        def _rank_value(row):
            ranks = rankdata(row, method='dense')
            return ranks / (ranks.max())

        # add rating and star rank
        df['ratings'] = df['ratings'].apply(_rank_value)
        df['stars'] = df['stars'].apply(_rank_value)

        # before padding we record the original impression impression str value
        # not drop them for target encoding purposes
        def ints2str(imps):
            if type(imps) != list:
                return np.nan
            else:
                return '|'.join([str(i) for i in imps])
        df['impressions_str'] = df['impressions'].apply(ints2str)

        # padding
        df['n_imps'] = df['impressions'].str.len()
        padding_mask = df['n_imps'] < 25
        # pad zeros for length less than 25
        cols_to_pad = ['impressions', 'prev_click', 'prev_interact']
        df = padding(df, pad_mask=padding_mask, cols_to_pad=cols_to_pad, padding_len=25, padding_value=np.nan)

        # pad ratings and stars
        cols_to_pad = ['ratings', 'stars']
        df = padding(df, pad_mask=padding_mask, cols_to_pad=cols_to_pad, padding_len=25, padding_value=np.nan)

        logger.debug(f'Nans:\n{df.isna().sum()}')

        # get target
        if mode == 'train':
            logger.info('Assign target in train mode, first convert reference id to int')
            # non-integer reference values
            non_int = df['reference'].str.contains(r'[^\d]+')
            n_non_int = non_int.sum()
            if n_non_int != 0:
                logger.warning(f'There are {n_non_int} number of non-int references')
                df.loc[non_int, 'reference'] = '-1'

            # nan reference values
            nan_ref = df['reference'].isna()
            n_nan_ref = nan_ref.sum()
            if n_nan_ref != 0:
                logger.warning(f'There are {n_nan_ref} number of nan references, drop them')
                logger.warning(f'{df[nan_ref]["session_id"].head()}')
                df = df[~nan_ref].reset_index(drop=True)

            # now convert to int
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
            logger.info(f'There are {nan_target.sum()} number of nan target '
                        '(reference id is not present in impressions)')
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

        # def _normalize(ps):
        #     p_arr = np.array(ps)
        #     return p_arr / (p_arr.max())
        # # normalize within
        # df['prices'] = df['prices'].apply(_normalize)
        logger.info('Add mean and median of prices')

        # def _add_mean_median(prices):
        #     return pd.Series([np.mean(prices), np.median(prices)])
        df[['mean_price', 'median_price', 'std_price']] = df.apply(lambda row: _add_mean_median_std(row['prices']),
                                                                   axis=1)

        # logger.info('Add price rank')
        # # add price rank
        # df['prices_rank'] = df['prices'].apply(_rank_value)
        #
        # # add first half page (first 12) price rank
        # def _rank_half_price(prices_row):
        #     # rank first 12 prices
        #     ranks = rankdata(prices_row[:12], method='dense')
        #     return ranks / (ranks.max())
        #
        # # add price rank
        # df['half_prices_rank'] = df['prices'].apply(_rank_half_price)

        # add price quantile instead
        def get_bins(prices):
            # get 5 bins
            prices_len = len(set(prices))
            n_bins = 1 if prices_len == 1 else 5  # use 5 bins
            return list(pd.qcut(prices, n_bins, duplicates='drop').codes+1)  # +1 so the index code start from 1

        logger.info('Getting price bins')
        df['price_bin'] = df['prices'].apply(get_bins)

        logger.info('Pad 0s for prices length shorter than 25 in prices and price_bin')
        padding_mask = df['prices'].str.len() < 25
        df = padding(df, pad_mask=padding_mask, cols_to_pad=['prices', 'price_bin'], padding_len=25,
                     padding_value=padding_value)

        # # pad the half page price rank
        # half_padding_mask = df['half_prices_rank'].str.len() < 12
        # df = padding(df, pad_mask=half_padding_mask, cols_to_pad=['half_prices_rank'],
        #              padding_len=12, padding_value=padding_value)

        # current_filters ==============================================================================================
        logger.info('Lower case current filters and split to list')
        df['current_filters'] = df['current_filters'].str.lower().str.split('|')
        df['n_cfs'] = df['current_filters'].str.len()
        df['n_cfs'].fillna(0, inplace=True)
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
        # drop_cols = ['reference']
        df.drop(drop_cols, axis=1, inplace=True)

        # finally expand all column of list to columns
        # expand_cols = ['prices', 'prices_rank', 'last_click', 'prev_click', 'last_interact',
        #                'prev_interact', 'current_filters']
        # expand_cols = ['prices', 'prices_rank', 'prev_click', 'prev_interact', 'current_filters',
        #                'half_prices_rank', 'ratings', 'stars']
        expand_cols = ['prices', 'price_bin', 'prev_click', 'prev_interact', 'current_filters',
                       'ratings', 'stars']
        for col in expand_cols:
            logger.info(f'Expanding on {col}')
            df = expand(df, col)

        # make sure categorical columns are int
        cat_cols = [c for c in df.columns if c in CATEGORICAL_COLUMNS]
        # there are nans in fs and sort_order
        df['fs'] = df['fs'].fillna(n_unique_fs)
        df['sort_order'] = df['sort_order'].fillna(n_unique_cs)
        df['last_action_type'] = df['last_action_type'].fillna(n_unique_actions)
        for c in cat_cols:
            df[c] = df[c].astype(int)

        # if test, only need to keep last row
        if mode == 'test':
            df = df.groupby('session_id').last().reset_index(drop=True)
        # save
        df.to_parquet(filename, index=False)
        logger.info(f'Total {mode} data input creation took: {(time.time()-t_init)/60:.2f} mins')
    logger.info('=' * 60)
    return df


if __name__ == '__main__':
    args = {'mode': 'train',
            'nrows': 100000,
            'add_test': False,
            'padding_value': np.nan,
            'recompute': True}

    logger.info(f'Creating model input: {args}')
    _ = create_model_inputs(**args)
