import numpy as np
from functools import partial
from utils import Fprint



# 0)
def remove_duplicates(df):
    # find duplciates except steps
    df.sort_values(by=['user_id', 'session_id', 'timestamp', 'step'],
                   ascending=[True, True, True, True],
                   inplace=True)
    duplicated_mask = df[[c for c in df.columns if c != 'step']].duplicated(keep='last')
    print('Before dropping duplicates df shape:', df.shape)
    df = df[~duplicated_mask].reset_index(drop=True)
    print('After dropping duplicates df shape:', df.shape)
    return df


# 1) Cliping sessions up to last clickout (if there is clickout)
def clip_last_click(grp):
    check = grp.action_type.values == 'clickout item'
    if check.sum() != 0:
        return grp.iloc[:np.argwhere(check)[-1][0]+1]
    else:
        return grp


# 2) Only select sessions that have a click out
def filter_clickout(grp, data_source='train'):
    # sessions has clickouts
    has_clickout = (grp['action_type'].values == 'clickout item').sum() != 0
    if data_source == 'train':
        # last row has reference and it's not nan
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['impressions', 'reference', 'prices']].isna().sum() == 0))
    else:
        # test should have the last reference as nan for clickout
        has_ref = ((grp['action_type'].iloc[-1] == 'clickout item') &
                   (grp.iloc[-1][['reference']].isna()))
    return has_clickout & has_ref


def preprocess_sessions(df, data_source='train', rd=True, fprint=None):
    if fprint is None:
        fprint = Fprint().fprint
    if rd:
        fprint('Remove initial duplciates')
        df = remove_duplicates(df)
    fprint('Cliping session dataframe up to last click out (if there is clickout)')
    df = df.groupby('session_id').apply(clip_last_click).reset_index(drop=True)

    fprint('filtering out sessions without clickouts, reference, or clickout is nan')
    print(f'{data_source} length before filtering: {len(df):,}')
    filter_func = partial(filter_clickout, data_source=data_source)
    valid_clicked = df.groupby('session_id').apply(filter_func)
    click_session_ids = valid_clicked[valid_clicked].index
    # filter
    df = df[df.session_id.isin(click_session_ids)].reset_index(drop=True)
    print(f'{data_source} length after filtering: {len(df):,}')
    return df
