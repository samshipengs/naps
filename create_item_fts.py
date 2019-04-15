import pandas as pd
import numpy as np
import time
from functools import partial
import multiprocessing as mp



# ====================================# ====================================# ====================================
def get_session_item_pairs(args):
    # grab the args
    gids, session_df, meta_df, actions_natural, data_source = args
    # selecting the assigned session ids and grouping on session level
    grps = (session_df[session_df['session_id'].isin(gids)]
            .reset_index(drop=True)
            .groupby('session_id'))

    #      meta_df= create_meta_fts(data_source)
    # use apply to compute session level features
    session_compute_func = partial(compute_session_item_pair,
                                   meta_df=meta_df,
                                   actions_natural=actions_natural,
                                   data_source=data_source)
    session_features = grps.apply(session_compute_func)

    return session_features


def compute_session_item_pair(session_df, meta_df, actions_natural, data_source):
    sdf = session_df.copy()
    # get last row and the rows above
    last_row = sdf.iloc[-1]
    above = sdf.iloc[:-1]
    # get previous appeared impressions
    prev = above[above['impressions'].notnull()]
    prev_imps = prev['impressions'].str.split('|')
    imps = [j for i in prev_imps for j in i]
    prev_imps_ctn = pd.value_counts(imps).to_dict()
    # get previous appeared clickout item_id counts
    prev_cos = prev['reference'].values
    prev_cos_ctn = pd.value_counts(prev_cos).to_dict()
    # get last row
    imp_l = last_row['impressions'].split('|')
    prices = last_row['prices'].split('|')
    prices = [int(p) for p in prices]
    # how many times the item_id has appeared before
    appeared = [int(prev_imps_ctn[i]) if i in imps else 0 for i in imp_l]
    # how many times the item_id was clicked before
    appeared_co = [int(prev_cos_ctn[int(i)]) if int(i) in prev_cos else 0 for i in imp_l]
    # the location of the impression
    locs = list(range(len(imp_l)))

    # previous occured action_types
    actions_above = above['action_type'].dropna().values
    actions_above_ohe = np.eye(10, dtype=int)[actions_above].sum(axis=0)

    # build the df
    result = pd.DataFrame({'appeared': appeared, 'appeared_co': appeared_co, 'location': locs, 'price': prices},
                          index=imp_l)
    # keep the time for split
    result['ts'] = last_row['timestamp']
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
    result.index = result.index.astype(int)
    result = result.join(meta_df, on='item_id')
    #     result['p_mean'] = np.mean(result['n_clicks'].values)
    result['star_mean'] = np.mean(result['star'].values)
    result['gr_mean'] = np.mean(result['good_rating'].values)
    result['sr_mean'] = np.mean(result['satisfactory_rating'].values)
    result['er_mean'] = np.mean(result['excellent_rating'].values)

    # add number of matched filters
    cfilter = last_row['current_filters']
    mfilter = result['properties']
    if pd.notna(cfilter):  # and pd.notna(mfilter):
        cfilters = cfilter.split('|')
        mfilters = mfilter.str.split('|')
        result['n_matches'] = [len(set(cfilters).intersection(p)) if type(p) != float else np.nan for p in mfilters]
        result['n_matches_per'] = result['n_matches'] / len(cfilter)
    else:
        result['n_matches'] = np.nan
    # reset index
    result.reset_index(inplace=True)

    if data_source == 'train':
        # get target
        ref = int(last_row['reference'])
        result['target'] = (result['item_id'].values == ref).astype(int)
    return result


# def generate_session_item_pairs(data_source, sessions_df, meta_df, nprocs=None):
def generate_session_item_pairs(data_source, sessions_df, meta_df, mapper_dict, nprocs=None):
    t1 = time.time()
    if nprocs is None:
        nprocs = mp.cpu_count() - 1
        print('Using {} cores'.format(nprocs))

    # get all the
    sids = sessions_df['session_id'].unique()

    actions_natural = mapper_dict['action_type']
    pairs = []

    # create iterator to pass in args
    def args_gen():
        for i in range(nprocs):
            yield (sids[range(i, len(sids), nprocs)], sessions_df, meta_df, actions_natural, data_source)

    # init multiprocessing pool
    pool = mp.Pool(nprocs)
    for pair in pool.map(get_session_item_pairs, args_gen()):
        pairs.append(pair)
    pool.close()
    pool.join()
    print('Done genearting, total time took: {0:.2f}mins'.format((time.time() - t1) / 60))

    return pd.concat(pairs, axis=0)
