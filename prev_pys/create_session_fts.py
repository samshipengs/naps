"""
Create session related features, such as number of unique actions, last reference id before last clickout etc.
"""
import numpy as np
import gc


def create_session_fts(df):
    # define some aggs
    session_aggs = {'timestamp': [span, mean_dwell_time, var_dwell_time, median_dwell_time, dwell_time_before_last],
                    'step': ['max'],
                    'action_type': ['nunique', n_clickouts, click_rel_pos_avg, second_last],
                    'reference': [second_last],
                    'city': ['nunique', get_last],
                    'platform': [get_last],
                    'device': [get_last],
                    'n_imps': [get_last],
                    'n_filters': [get_last],
                    }

    df['imp_list'] = df.impressions.str.split('|')
    df['n_imps'] = df.imp_list.str.len()
    del df['imp_list']
    gc.collect()
    df['cf_list'] = df.current_filters.str.split('|')
    df['n_filters'] = df.cf_list.str.len()
    del df['cf_list']
    gc.collect()

    session_grp = df.groupby('session_id')
    session_fts = session_grp.agg(session_aggs)
    session_fts.columns = ['_'.join(col).strip() for col in session_fts.columns.values]
    del df['n_imps'], df['n_filters']
    gc.collect()
    return session_fts


# some custom funcs used in agggregation
def mean_dwell_time(x):
    if len(x) == 1:
        return np.nan
    else:
        return np.mean(np.diff(np.sort(x)))


def median_dwell_time(x):
    if len(x) == 1:
        return np.nan
    else:
        return np.median(np.diff(np.sort(x)))


def dwell_time_before_last(x):
    if len(x) == 1:
        return np.nan
    else:
        sorted_x = np.sort(x)
        return sorted_x[-1] - sorted_x[-2]


def var_dwell_time(x):
    if len(x) == 1:
        return np.nan
    else:
        return np.var(np.diff(np.sort(x)))


def get_first(x):
    return x.iloc[0]


def get_last(x):
    return x.iloc[-1]


def n_clickouts(x):
    # 'clickout item': 2
    return (x.values == 2).sum()


def click_rel_pos_avg(x):
    # 'clickout item': 2
    return np.mean(np.argwhere((x.values == 2)) + 1) / len(x)


def span(x):
    return x.max() - x.min()


def second_last(x):
    if len(x) == 1:
        return np.nan
    else:
        return x.iloc[-2]

