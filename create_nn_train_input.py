import pandas as pd
import numpy as np
import datetime, os, gc
from utils import load_data, get_logger, check_dir
from clean_session import preprocess_sessions


logger = get_logger('create_nn_input')


def flogger(df, name):
    logger.info(f'{name} shape: ({df.shape[0]:,}, {df.shape[1]})')


def load_train(nrows):
    # first load data
    train = load_data('train', nrows=nrows)

    # preprocess data i.e. Dropping duplicates, only take sessions with clicks and clip to last click out
    train = preprocess_sessions(train, recompute=True)

    # get time and select columns that get used
    train['timestamp'] = train['timestamp'].apply(lambda t: datetime.datetime.utcfromtimestamp(t))
    usecols =['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'current_filters',
              'reference', 'impressions', 'prices']
    train = train[usecols]
    flogger(train, 'train')
    return train


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
    if mask.sum() <= 1:
        return np.nan
    else:
        return rids[mask].iloc[-2]  # the second last i.e. the one before click out


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


def create_train_inputs(nrows=100000, recompute=False):
    filepath = './cache'
    check_dir(filepath)
    filenames = [os.path.join(filepath, f'{i}.npy') for i in ['train_numerics', 'train_impressions', 'train_prices',
                                                              'train_cfilters', 'train_targets']]
    if sum([os.path.isfile(f) for f in filenames]) == len(filenames) and not recompute:
        logger.info(f'LOAD FROM EXISTING {filenames}')
        # train = pd.read_parquet(filename)
        # train = pd.read_hdf(filename, 'train')
        numerics, impressions, prices, cfilters, targets = [np.load(f) for f in filenames]
    else:
        logger.info('LOAD TRAIN')
        train = load_train(nrows)
        logger.info('COMPUTE SESSION FEATURES')
        train = compute_session_fts(train)

        logger.info('ONLY SELECT LAST CLICKOUT FROM EACH SESSION')
        train = train.groupby('session_id').last().reset_index()

        logger.info('LOWER CASE CURRENT FILTERS AND SPLIT TO LIST')
        train['cfs'] = train['current_filters_last_filters'].str.lower().str.split('|')

        logger.info('SPLIT PRICES STR TO LIST AND CONVERT TO INT')
        train['prices'] = train['prices'].str.split('|')
        train['prices'] = train['prices'].apply(lambda x: [int(p) for p in x])
        logger.info('PAD 0S FOR LENGTH SHORTER THAN 25 (HMM)')
        train['prices'] = train.prices.apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))

        logger.info('SPLIT IMPRESSION STR TO LIST OF IMPRESSIONS')
        train['impressions'] = train['impressions'].str.split('|')
        logger.info('CONVERT IMPRESSION STR TO INT')
        train['impressions'] = train['impressions'].apply(lambda x: [int(i) for i in x])
        logger.info('CONVERT REFERENCE ID TO INT')
        train['reference'] = train['reference'].astype(int)
        logger.info('PAD 0S FOR LENGTH SHORTER THAN 25 (HMM)')
        train['impressions'] = train['impressions'].apply(lambda x: np.pad(x, (0, 25-len(x)), mode='constant'))

        logger.info('ASSIGN TARGET')

        # filter out nan rows with reference_id not in impressions list, since if the true target in test
        # is not in the impression list then it would not get evaluated
        def assign_target(row):
            ref = row['reference']
            imp = list(row['impressions'])
            if ref in imp:
                return imp.index(ref)
            else:
                return np.nan

        logger.info('ALSO ASSIGN LOCATION OF PREVIOUS REFERENCE ID')

        def assign_last_ref_id(row):
            ref = row['reference_last_reference_id']
            # imp = [str(i) for i in row['impressions']]
            imp = list(row['impressions'])
            if pd.isna(ref):
                return np.nan
            else:
                if ref in imp:
                    return (imp.index(ref)+1)/len(imp)
                else:
                    return np.nan

        train['target'] = train.apply(assign_target, axis=1)
        train['last_ref_ind'] = train.apply(assign_last_ref_id, axis=1)

        logger.info('REMOVE THE ONES WHOSE REFERENCE ID IS NOT IN IMPRESSION LIST')
        # drop the ones whose reference is not in the impression list
        train = train[train['target'].notna()].reset_index(drop=True)
        train['target'] = train['target'].astype(int)

        logger.info(f"Look at target distribution: \n{pd.value_counts(train['target']).head()}")

        # create meta ohe
        logger.info('LOAD META DATA')
        meta_df = load_data('item_metadata')
        logger.info('LOWER AND SPLIT PROPERTIES STR TO LIST')
        meta_df['properties'] = meta_df['properties'].str.lower().str.split('|')
        logger.info('GET ALL UNIQUE PROPERTIES')
        unique_properties = list(set(np.concatenate(meta_df['properties'].values)))
        property2natural = {v: k for k, v in enumerate(unique_properties)}
        n_properties = len(unique_properties)
        logger.info(f'TOTAL NUMBER OF UNIQUE META PROPERTIES: {n_properties}')
        logger.info('CONVERT THE PROPERTIES TO OHE AND SUPERPOSE FOR EACH ITEM_ID')
        meta_df['properties'] = meta_df['properties'].apply(lambda ps: [property2natural[p] for p in ps])
        meta_df['properties'] = meta_df['properties'].apply(lambda ps: np.sum(np.eye(n_properties, dtype=int)[ps], axis=0))
        logger.info('CREATE MAPPINGS')
        meta_mapping = dict(meta_df[['item_id', 'properties']].values)
        logger.info('SAVING META MAPPING')
        np.save('./cache/meta_mapping.npy', meta_mapping)
        # add a mapping for the padded values
        meta_mapping[0] = np.zeros(n_properties, dtype=int)
        del meta_df, unique_properties, property2natural
        gc.collect()

        logger.info('APPLY META OHE MAPPING TO IMPRESSIONS (THIS COULD TAKE SOME TIME)')
        train['impressions'] = (train['impressions']
                                .apply(lambda imps: np.vstack([meta_mapping[i]
                                                               if i in meta_mapping.keys()
                                                               else np.zeros(n_properties, dtype=int)
                                                               for i in imps])))
        del meta_mapping
        gc.collect()

        logger.info('CREATE OHE SUPERPOSITION OF FILTERS')
        unique_cfs = list(set(np.concatenate(train['cfs'].dropna().values)))
        cfs_mapping = {v: k for k, v in enumerate(unique_cfs)}
        logger.info('SAVING FILTERS MAPPING')
        np.save('./cache/filters_mapping.npy', cfs_mapping)
        n_cfs = len(unique_cfs)
        logger.info(f'THERE ARE TOTAL {n_cfs} UNIQUE FILTERS')

        logger.info('APPLY FILTERS OHE SUPERPOSITION TO EACH RECORDS')
        train.loc[train['cfs'].notna(), 'cfs'] = (train.loc[train['cfs'].notna(), 'cfs']
                                                  .apply(lambda cfs: [cfs_mapping[cf] for cf in cfs]))
        # zeros if the cfs is nan (checked with type(cfs) is list not float
        train['cfs'] = (train['cfs'].apply(lambda cfs: np.sum(np.eye(n_cfs, dtype=int)[cfs], axis=0)
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

        train['prices'] = train['prices'].apply(normalize)
        # PRICES
        prices = np.array(list(train['prices'].values))
        del train['prices']
        save_cache(prices, 'train_prices.npy')

        logger.info('Getting impressions')
        # IMPRESSIONS
        impressions = np.array(list(train['impressions'].values))
        del train['impressions']
        save_cache(impressions, 'train_impressions.npy')

        logger.info('Getting current_filters')
        # CURRENT_FILTERS
        cfilters = np.array(list(train['cfs'].values))
        del train['cfs']
        save_cache(cfilters, 'train_cfilters.npy')

        logger.info('Getting numerics')
        # numerics
        num_cols = ['session_id_size', 'timestamp_dwell_time_prior_clickout', 'last_ref_ind']
        for c in num_cols:
            train[c] = train[c].fillna(-1)
        numerics = train[num_cols].values
        train = train.drop(num_cols, axis=1)
        save_cache(numerics, 'train_numerics.npy')

        logger.info('Getting targets')
        # TARGETS
        targets = train['target'].values
        del train['target']
        save_cache(targets, 'train_targets.npy')

    return numerics, impressions, prices, cfilters, targets


def pipeline():
    _ = create_train_inputs(nrows=1000000)





if __name__ == '__main__':
    pipeline()