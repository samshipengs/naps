import gc
import os
import numpy as np
import pandas as pd
from utils import Fprint, check_dir


def session_duration(ts):
    """
    Total session duration
    """
    if len(ts) == 1:
        return np.nan
    else:
        return ts.max() - ts.min()


def mean_dwell_time(ts):
    """
    Average dwell time
    """
    if len(ts) == 1:
        return np.nan
    else:
        return np.mean(np.diff(np.sort(ts)))


def median_dwell_time(ts):
    """
    Median dwell time
    """
    if len(ts) == 1:
        return np.nan
    else:
        return np.median(np.diff(np.sort(ts)))


def var_dwell_time(ts):
    """
    Variance of dewell time
    """
    if len(ts) == 1:
        return np.nan
    else:
        return np.var(np.diff(np.sort(ts)))


def dwell_time_prior_clickout(ts):
    """
    Duration before last clickout
    """
    if len(ts) == 1:
        return np.nan
    else:
        sorted_ts = np.sort(ts)
        return sorted_ts[-1] - sorted_ts[-2]


def dwell_time_prior_clickout_per(ts):
    """
    Percentage of last interaction dwell time before clickout
    """
    if len(ts) == 1:
        return np.nan
    else:
        sorted_ts = np.sort(ts)
        assert (sorted_ts[-1] - sorted_ts[0]) != 0, f'should not equal{sorted_ts} | {ts.index}'
        return (sorted_ts[-1] - sorted_ts[-2]) / (sorted_ts[-1] - sorted_ts[0])


def second_last(x):
    """
    Get the seoncd last value
    """
    if len(x) == 1:
        return np.nan
    else:
        return x.iloc[-2]


def n_prev_clickouts(action_type):
    """
    Count how many previous clickouts are there
    """
    return (action_type.values == 'clickout item').sum() - 1


def avg_clickout_loc(action_type):
    """
    Average clickout location in the session
    """
    return np.mean(np.argwhere((action_type.values == 'clickout item')) + 1) / len(action_type)


def compute_session_fts(df, data_source='train'):
    """
    Create session features using groupby with agg
    """
    fprint = Fprint().fprint
    filepath = './cache'
    check_dir(filepath)
    filename = os.path.join(filepath, 'session_fts.h5')
    if os.path.isfile(filename):
        store = pd.HDFStore(filename)
        if data_source in store.keys():
            fprint(f'Load {data_source} from existing {filename}')
            session_fts = pd.read_hdf(filename, data_source)
            return session_fts
    # define some aggs
    session_aggs = {'timestamp': [session_duration, mean_dwell_time, var_dwell_time, median_dwell_time,
                                  dwell_time_prior_clickout, dwell_time_prior_clickout_per],
                    'step': ['max'],
                    'action_type': ['nunique', n_prev_clickouts, avg_clickout_loc, second_last],
                    'reference': ['nunique', second_last],
                    'city': ['last'],
                    'platform': ['last'],
                    'device': ['last'],
                    # below opeartes on createed features
                    'n_imps': ['last'],
                    'n_filters': ['last']}

    fprint("Generate length of 'impressions' and 'current_filters'")
    df['n_imps'] = df.impressions.str.split('|').str.len()
    df['n_filters'] = df.current_filters.str.split('|').str.len()

    fprint("Creating session features using agg on groupby from 'session_id'")
    session_grp = df.groupby('session_id')
    session_fts = session_grp.agg(session_aggs)

    fprint('Done creating session fts, cleaning up column names')
    session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
    del df['n_imps'], df['n_filters']
    gc.collect()

    fprint('Reset session_fts index and save it to h5')
    session_fts.reset_index(inplace=True)
    session_fts.to_hdf(filename, data_source)
    return session_fts
