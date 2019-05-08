import pandas as pd
import numpy as np
import datetime, os, gc
from utils import load_data, get_logger, check_dir
from clean_session import preprocess_sessions
from create_nn_train_input import create_cfs_mapping


logger = get_logger('create_nn_input')


def flogger(df, name):
    logger.info(f'{name} shape: ({df.shape[0]:,}, {df.shape[1]})')


def load_test(nrows=None):
    # first test load data
    test = load_data('test', nrows=nrows)
    flogger(test, 'raw test')
    # then load the test that we need to submit
    test_sub = load_data('submission_popular')
    sub_sids = test_sub['session_id'].unique()
    test = test[test['session_id'].isin(sub_sids)].reset_index(drop=True)
    flogger(test, 'test with only ids needed for submissions')

    # pre-process data i.e. Dropping duplicates, only take sessions with clicks and clip to last click out
    test = preprocess_sessions(test, data_source='test', recompute=True)

    # get time and select columns that get used
    test['timestamp'] = test['timestamp'].apply(lambda t: datetime.datetime.utcfromtimestamp(t))
    usecols = ['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'current_filters',
               'reference', 'impressions', 'prices']
    test = test[usecols]
    flogger(test, 'test')
    return test


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


def last_filters(cf):
    mask = cf.notna()
    if mask.sum() == 0:
        return np.nan
    else:
        return cf[mask].iloc[-1]


def last_reference_id(rids):
    mask = rids.notna()
    if mask.sum() == 0:
        return np.nan
    else:
        return rids[mask].iloc[-1]
        # the last i.e. the one before click out (for test as test does not have reference
        # id for last row)


def compute_session_fts(df):
    aggs = {'timestamp': [session_duration, dwell_time_prior_clickout],
            'current_filters': [last_filters],
            'session_id': 'size',
            'reference': [last_reference_id]}
    session_grp = df.groupby('session_id')
    session_fts = session_grp.agg(aggs)
    session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
    session_fts.reset_index(inplace=True)

    return pd.merge(df, session_fts, on='session_id')


def save_cache(arr, name, filepath='./cache'):
    np.save(os.path.join(filepath, name), arr)


def create_test_inputs(nrows=None, recompute=False):
    filepath = './cache'
    check_dir(filepath)
    filenames = [os.path.join(filepath, f'{i}.npy') for i in ['test_numerics', 'test_impressions', 'test_prices',
                                                              'test_cfilters']]
    if sum([os.path.isfile(f) for f in filenames]) == len(filenames) and not recompute:
        logger.info(f'LOAD FROM EXISTING {filenames}')
        # test = pd.read_parquet(filename)
        # test = pd.read_hdf(filename, 'test')
        numerics, impressions, prices, cfilters = [np.load(f) for f in filenames]
    else:
        logger.info('LOAD TEST')
        test = load_test(nrows)
        logger.info('COMPUTE SESSION FEATURES')
        test = compute_session_fts(test)

        logger.info('ONLY SELECT LAST CLICKOUT FROM EACH SESSION')
        test = test.groupby('session_id').last().reset_index()

        test_sub = test[['session_id', 'impressions']]
        test_sub.to_csv('./cache/test_sub.csv')
        del test_sub

        logger.info('LOWER CASE CURRENT FILTERS AND SPLIT TO LIST')
        test['cfs'] = test['current_filters_last_filters'].str.lower().str.split('|')

        logger.info('SPLIT PRICES STR TO LIST AND CONVERT TO INT')
        test['prices'] = test['prices'].str.split('|')
        test['prices'] = test['prices'].apply(lambda x: [int(p) for p in x])
        logger.info('PAD 0S FOR LENGTH SHORTER THAN 25 (HMM)')
        test['prices'] = test.prices.apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))

        logger.info('SPLIT IMPRESSION STR TO LIST OF IMPRESSIONS')
        test['impressions'] = test['impressions'].str.split('|')

        logger.info('CONVERT IMPRESSION STR TO INT')
        test['impressions'] = test['impressions'].apply(lambda x: [int(i) for i in x])
        # logger.info('CONVERT REFERENCE ID TO INT')
        # test['reference'] = test['reference'].astype(int)
        logger.info('PAD 0S FOR LENGTH SHORTER THAN 25 (HMM)')
        test['impressions'] = test['impressions'].apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))

        logger.info('ASSIGN LOCATION OF PREVIOUS REFERENCE ID')

        def assign_last_ref_id(row):
            ref = row['reference_last_reference_id']
            imp = [str(i) for i in row['impressions']]
            if pd.isna(ref):
                return np.nan
            else:
                if ref in imp:
                    return imp.index(ref) + 1
                    # return (imp.index(ref)+1)/len(imp)
                else:
                    return np.nan

        test['last_ref_ind'] = test.apply(assign_last_ref_id, axis=1)

        # create meta ohe
        meta_mapping = np.load('./cache/meta_mapping.npy').item()
        n_properties = len(meta_mapping[list(meta_mapping.keys())[0]])

        logger.info('APPLY META OHE MAPPING TO IMPRESSIONS (THIS COULD TAKE SOME TIME)')
        test['impressions'] = (test['impressions']
                               .apply(lambda imps: np.vstack([meta_mapping[i]
                                                              if i in meta_mapping.keys()
                                                              else np.zeros(n_properties, dtype=int)
                                                              for i in imps])))
        del meta_mapping
        gc.collect()

        # cfs_mapping = np.load('./cache/filters_mapping.npy').item()
        # n_cfs = len(cfs_mapping)
        cfs_mapping, n_cfs = create_cfs_mapping()
        logger.info(f'THERE ARE TOTAL {n_cfs} UNIQUE FILTERS')

        logger.info('APPLY FILTERS OHE SUPERPOSITION TO EACH RECORDS')
        test.loc[test['cfs'].notna(), 'cfs'] = (test.loc[test['cfs'].notna(), 'cfs']
                                                .apply(lambda cfs: [cfs_mapping[cf] for cf in cfs]))
        # zeros if the cfs is nan (checked with type(cfs) is list not float
        test['cfs'] = (test['cfs'].apply(lambda cfs: np.sum(np.eye(n_cfs, dtype=int)[cfs], axis=0)
                                         if type(cfs) == list else np.zeros(n_cfs, dtype=int)))
        del cfs_mapping
        gc.collect()

        logger.info('Grabbing list of inputs')
        logger.info('Normalizing price')

        # maybe normalize to percentage within each records, check does each item_id have the same price
        # over all records
        def normalize(ps):
            p_arr = np.array(ps)
            return p_arr / (p_arr.max())

        test['prices'] = test['prices'].apply(normalize)
        # PRICES
        prices = np.array(list(test['prices'].values))
        del test['prices']
        save_cache(prices, 'test_prices.npy')

        logger.info('Getting impressions')
        # IMPRESSIONS
        impressions = np.array(list(test['impressions'].values))
        del test['impressions']
        save_cache(impressions, 'test_impressions.npy')

        logger.info('Getting current_filters')
        # CURRENT_FILTERS
        cfilters = np.array(list(test['cfs'].values))
        del test['cfs']
        save_cache(cfilters, 'test_cfilters.npy')

        logger.info('Getting numerics')
        # numerics
        num_cols = ['session_id_size', 'timestamp_dwell_time_prior_clickout', 'last_ref_ind']
        for c in num_cols:
            test[c] = test[c].fillna(-1)
        numerics = test[num_cols].values
        test = test.drop(num_cols, axis=1)
        save_cache(numerics, 'test_numerics.npy')

    return numerics, impressions, prices, cfilters


def pipeline():
    _ = create_test_inputs()


if __name__ == '__main__':
    pipeline()
