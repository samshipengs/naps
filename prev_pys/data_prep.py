import datetime
import time 
import gc
import multiprocessing as mp
from functools import partial
from ast import literal_eval


from utils import *


def action_encoding():
    # encode action type info into reference ids, we do so by getting all the data from both train and test
    action_train = load_data('train', usecols=['action_type', 'reference'])
    action_test = load_data('test', usecols=['action_type', 'reference'])
    action_cb = pd.concat([action_train, action_test], ignore_index=True)
    del action_train, action_test
    # remove rows whose action type is not ids
    no_ids = ['search for poi', 'search for destination', 'filter selection', 'change of sort order']
    action_cb = action_cb[~action_cb['action_type'].isin(no_ids)].reset_index(drop=True)
    action_cb = action_cb.dropna().reset_index(drop=True)
    is_id = action_cb['reference'].str.match('\d')
    action_cb = action_cb[is_id].reset_index(drop=True)
    action_cb['reference'] = action_cb['reference'].astype(int)
    # now group on reference ids
    action_grp = action_cb.groupby('reference')['action_type']
    # get value counts for each action type
    action_ctn = action_grp.value_counts()
    # list of all unique action type
    actions = list(action_cb.action_type.unique())
    # create ohe
    action_mapping = {v: k for k, v in enumerate(actions)}
    action_ctn_df = action_ctn.reset_index(name='ctn')
    action_ctn_df['action_id'] = action_ctn_df['action_type'].map(action_mapping)
    # set this into a df
    ohe = pd.DataFrame(np.eye(len(actions), dtype=int)[action_ctn_df.action_id.values], columns=actions)
    ohe = ohe.mul(action_ctn_df['ctn'], axis=0)
    action_ctn_df = pd.concat([action_ctn_df, ohe], axis=1)

    action_encoding = action_ctn_df.groupby('reference')[actions].sum()
    # also add normalized percentage over count of each actions over total
    normalized = action_encoding.div(action_encoding.sum(axis=1), axis=0)
    action_encoding = action_encoding.join(normalized, lsuffix='_ctn', rsuffix='_per')
    return action_encoding.reset_index()


# get all rows upto the last clickout action (some rows has reference but it's not clickout action)
def up_to_last_click(grp):
    check = grp.action_type == 'clickout item'
    if check.sum() != 0:
        return grp.iloc[:np.argwhere(check)[-1][0]+1]
    else:
        return grp


# only look at sessions with clickouts (for now)
# first filter out sessions that does not have a click-out
def check_clickout(grp, mode):
     # sessions has clickouts
    has_clickout = 'clickout item' in grp['action_type'].unique()
    if mode == 'train':
        # last row has reference and it's not nan
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['impressions', 'reference', 'prices']].isna().sum() == 0))
    else:
        # test should have the last reference as nan for clickout
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['reference']].isna()))
    return has_clickout & has_ref


def create_rating_colum(meta):
    # create rating columns
    ratings = ['good rating', 'satisfactory rating', 'excellent rating']
    for r in ratings:
        meta[r.replace(' ', '_')] = meta.properties.str.findall(f'\|{r}').str.len()


def get_popularity():
    """Get number of clicks that each item received in the df."""
    # encode action type info into reference ids, we do so by getting all the data from both train and test
    action_train = load_data('train', usecols=['action_type', 'reference'])
    action_test = load_data('test', usecols=['action_type', 'reference'])
    df = pd.concat([action_train, action_test], ignore_index=True)
    del action_train, action_test

    mask = df["action_type"] == "clickout item"
    df_clicks = df[mask]
    df_item_clicks = (df_clicks
                      .groupby("reference")
                      .size()
                      .reset_index(name="n_clicks")
                      .transform(lambda x: x.astype(int)))
    return df_item_clicks


# ====================================# ====================================# ====================================


def create_session_fts(data_source, train_df=None, recompute=False):
    # import os.path
    session_file = f'./data/{data_source}_session_fts.csv'
    if os.path.isfile(session_file) and not recompute:
        print(f'{session_file} exists, reload')
        session_fts = pd.read_csv(session_file)
        session_fts.set_index('session_id', inplace=True)
    else:
        # define some aggs
        session_aggs = {'timestamp': [ptp, mean_dwell_time, var_dwell_time],
                        'step': ['max'],
                        'action_type': ['nunique', n_clickouts, click_rel_pos_avg],
                        'city': ['nunique', get_first],
                        'platform': [get_first],
                        'device': [get_first],
                        'nfilters': ['mean', 'max', 'min', get_last],
                        'nimps': ['max']
                        }
        session_grp = train_df.groupby('session_id')
        session_fts = session_grp.agg(session_aggs)
        session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
        session_fts.to_csv(session_file)

    return session_fts

# ====================================# ====================================# ====================================


def create_meta_fts(data_source, nrows=None, recompute=False):
    meta_file = f'./data/{data_source}_meta_fts.csv'
    if os.path.isfile(meta_file) and not recompute:
        print(f'{meta_file} exists, reload')
        meta = pd.read_csv(meta_file)
        # convert 'list' to list
        meta.loc[:, 'ps'] = meta.loc[:, 'ps'].apply(literal_eval)

        meta = meta.set_index('item_id')
    else:
        meta = load_data('item_metadata', nrows=nrows)
        print('add more columns to meta')
        meta['properties'] = meta['properties'].str.lower()
        meta['ps'] = meta['properties'].str.split('|')
        # numer of properties
        meta['nprop'] = meta.ps.str.len()
        # star ratings
        meta['star'] = meta.properties.str.extract('[\|](\d) star')
        meta['star'] = meta['star'].astype(float)
        # add ratings
        create_rating_colum(meta)
        # get popularity
        item_popularity = get_popularity()
        meta = pd.merge(meta, item_popularity, left_on='item_id', right_on='reference')
        del item_popularity
        gc.collect()
        # action encodings
        action_encodings = action_encoding()
        meta = pd.merge(meta, action_encodings, left_on='item_id', right_on='reference')
        # choose columns
        act_cols = [c for c in action_encodings.columns if c != 'reference']

        use_cols = ['item_id', 'nprop', 'n_clicks', 'star', 'good_rating', 'satisfactory_rating',
                    'excellent_rating', 'ps']
        use_cols += act_cols
        meta = meta[use_cols].set_index('item_id')
        meta.to_csv(meta_file)
    return meta


# ====================================# ====================================# ====================================
def action_type_mapping():
    return {'search for poi': 0, 'interaction item image': 1, 'clickout item': 2, 'interaction item info': 3,
            'interaction item deals': 4, 'search for destination': 5, 'filter selection': 6,
            'interaction item rating': 7, 'search for item': 8, 'change of sort order': 9}


actions_natural = action_type_mapping()


# ====================================# ====================================# ====================================
def get_session_item_pairs(args):
    # grab the args
    gids, session_df, data_source = args
    # selecting the assigned session ids and grouping on session level
    grps = (session_df[session_df['session_id'].isin(gids)]
            .reset_index(drop=True)
            .groupby('session_id'))

    meta_df = create_meta_fts(data_source)
    # use apply to compute session level features
    session_compute_func = partial(compute_session_item_pair, meta_df=meta_df, data_source=data_source)
    session_features = grps.apply(session_compute_func)

    return session_features


def compute_session_item_pair(session_df, meta_df, data_source):
    sdf = session_df.copy()
    last_row = sdf.iloc[-1]
    above = sdf.iloc[:-1]
    # get previous appeared impressions
    prev = above[above['impressions'].notnull()]
    prev_imps = prev['imps_list']
    imps = [j for i in prev_imps for j in i]
    prev_imps_ctn = pd.value_counts(imps).to_dict()

    # get previous appeared clickout item_id counts
    prev_cos = prev['reference'].values
    prev_cos_ctn = pd.value_counts(prev_cos).to_dict()

    # get last row
    imp_l = last_row['imps_list']
    prices = last_row['prices'].split('|')
    prices = [int(p) for p in prices]

    # how many times the item_id has appeared before
    appeared = [int(prev_imps_ctn[i]) if i in imps else 0 for i in imp_l]
    # how many times the item_id was clicked before
    appeared_co = [int(prev_cos_ctn[i]) if i in prev_cos else 0 for i in imp_l]

    # the location of the impression
    locs = list(range(len(imp_l)))

    # previous occured action_types
    actions_above = above['action_type'].dropna().map(actions_natural).values
    actions_above_ohe = np.eye(10, dtype=int)[actions_above].sum(axis=0)

    # build the df
    result = pd.DataFrame({'appeared': appeared, 'appeared_co': appeared_co, 'location': locs, 'price': prices},
                          index=imp_l)
    result.index.name = 'item_id'
    price_ind = np.argsort(result['price'].values) + 1
    result['rel_price_rank'] = price_ind / len(imp_l)

    result['price_mean'] = np.mean(result['price'])
    result['price_median'] = np.median(result['price'])

    result_price = result['price'].values
    result_price_mean = result['price_mean'].values
    result_price_median = result['price_median'].values

    result['diff_mean'] = result_price - result_price_mean
    result['diff_median'] = result_price - result_price_median
    result['diff_mean_rel'] = (result_price - result_price_mean) / result_price
    result['diff_median_rel'] = (result_price - result_price_median) / result_price

    # add previous action type cols (same for all rows)
    prev_act_cols = [f'prev_{i}' for i in actions_natural.keys()]
    result[prev_act_cols] = pd.DataFrame(np.tile(actions_above_ohe, (len(result), 1)), index=result.index)

    # fetch the meta data
    result = result.join(meta_df, on='item_id')
    result['p_mean'] = np.mean(result['n_clicks'].values)
    result['star_mean'] = np.mean(result['star'].values)
    result['gr_mean'] = np.mean(result['good_rating'].values)
    result['sr_mean'] = np.mean(result['satisfactory_rating'].values)
    result['er_mean'] = np.mean(result['excellent_rating'].values)

    # add number of matched filters
    cfilter = last_row['filters']
    mfilter = result['ps'].values
    if (type(cfilter) != float) and (type(mfilter) != float):
        result['n_matches'] = [len(set(cfilter).intersection(p)) if type(p) != float
                               else 0 for p in mfilter]
        result['n_matches_per'] = result['n_matches']/len(cfilter)
    else:
        result['n_matches'] = np.nan
    # do not need to return 'ps' column
    del result['ps']
    # reset index
    result.reset_index(inplace=True)

    if data_source == 'train':
        # get target
        ref = int(last_row['reference'])
        result['target'] = (result['item_id'].values == ref).astype(int)
    return result


# def generate_session_item_pairs(data_source, sessions_df, meta_df, nprocs=None):
def generate_session_item_pairs(data_source, sessions_df, nprocs=None):
    t1 = time.time()
    if nprocs is None:
        nprocs = mp.cpu_count() - 1
        nprocs = 11
        print('Using {} cores'.format(nprocs))

    sids = sessions_df['session_id'].unique()

    pairs = []

    # create iterator to pass in args
    def args_gen():
        for i in range(nprocs):
            yield (sids[range(i, len(sids), nprocs)], sessions_df, data_source)

    # init multiprocessing pool
    pool = mp.Pool(nprocs)
    for pair in pool.map(get_session_item_pairs, args_gen()):
        pairs.append(pair)
    pool.close()
    pool.join()
    print('Done genearting, total time took: {0:.2f}mins'.format((time.time() - t1) / 60))

    return pd.concat(pairs, axis=0)


def genearte_data(data_source='train', nrows=10000):
    assert data_source in ['train', 'test'], 'provide valid data source'
    t_int = time.time()
    fprint = lambda msg: print(f'{msg:<40} {"="*20} time elapsed = {(time.time()-t_int)/60:.2f} mins')

    fprint(f'Generating data for {data_source.upper()}')
    df = load_data(data_source, nrows=nrows)
    if data_source == 'test':
        print('Only load sessions from submission sample')
        sample = load_data('submission_popular', usecols=['session_id'])
        sample_ids = sample['session_id'].unique()
        df = df[df['session_id'].isin(sample_ids)].reset_index(drop=True)

    fprint('cliping sessions off up to last clickout')
    df = df.groupby('session_id').apply(up_to_last_click).reset_index(drop=True)

    fprint('get utc time')
    df['ts'] = df['timestamp'].apply(lambda t: datetime.datetime.utcfromtimestamp(t))

    fprint('filtering out sessions without clickout and reference for clickout is not valid')
    fprint(f'{data_source} length before filtering: {len(df):,}')
    check_clickout_ = partial(check_clickout, mode=data_source)
    clicked = df.groupby('session_id').apply(check_clickout_)
    click_session_ids = clicked[clicked].index
    # filter
    df = df[df.session_id.isin(click_session_ids)].reset_index(drop=True)
    print('='*30, 'save only sessions with clicks')
    df.to_csv('./test_click.csv', index=False)
    print('='*30, 'done saving')
    del clicked, click_session_ids
    gc.collect()
    fprint(f'{data_source} length after filtering: {len(df):,}')

    # add additional columns
    fprint('add more columns in train')
    df['current_filters'] = df['current_filters'].str.lower()
    df['filters'] = df['current_filters'].str.split('|')
    df['nfilters'] = df['filters'].str.len()
    # del df['filters']
    # gc.collect()

    df['imps_list'] = df['impressions'].str.split('|')
    nn_mask = df['imps_list'].notnull()
    df.loc[nn_mask, 'imps_list'] = df.loc[nn_mask, 'imps_list'].apply(lambda x: [int(i) for i in x])
    df['nimps'] = df['imps_list'].str.len()

    # session fts
    fprint('Get session features')
    session_fts = create_session_fts(data_source, df)
    del session_fts
    gc.collect()
    # meta fts
    fprint('Get meta features')
    meta = create_meta_fts(data_source)
    del meta
    gc.collect()

    # session_item pairs
    fprint('Get session item pair')
    # session_items = generate_session_item_pairs(data_source, df, meta, nprocs=None)
    session_items = generate_session_item_pairs(data_source, df, nprocs=None)
    # join
    session_items.reset_index(level='session_id', inplace=True)
    session_items.set_index('session_id', inplace=True)

    # make note of train and val split then remove train
    # split to train and valid
    train_sids = df[df['ts'] <= datetime.datetime(2018, 11, 6)].session_id.unique()
    # del df, meta
    del df
    gc.collect()

    # final
    session_fts = create_session_fts(data_source)
    final = session_items.join(session_fts)

    del session_fts, session_items
    gc.collect()

    # final.to_csv('./data/final.csv')
    if data_source == 'train':
        xtrain = final[final.index.isin(train_sids)]
        xval = final[~final.index.isin(train_sids)]
        xtrain.reset_index(inplace=True)
        xval.reset_index(inplace=True)
        fprint(f'Write xtrain {xtrain.shape} to h5')
        xtrain.to_hdf('./data/train.h5', key='xtrain', mode='w')
        fprint(f'Write xval {xval.shape} to h5')
        xval.to_hdf('./data/train.h5', key='xval', mode='a')
    else:
        final.reset_index(inplace=True)
        final.to_hdf('./data/test.h5', key='xtest', mode='w')

    fprint('Done generate data')


if __name__ == '__main__':
    data_source = 'train'
    genearte_data(data_source, nrows=None)
