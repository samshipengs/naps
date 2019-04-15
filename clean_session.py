import numpy as np
import gc
from functools import partial


# 1) Cliping sessions up to last clickout (if there is clickout)
def clip_last_click(grp):
    # 'clickout item': 2
    check = grp.action_type.values == 2
    if check.sum() != 0:
        return grp.iloc[:np.argwhere(check)[-1][0]+1]
    else:
        return grp


# 2) Only select sessions that have a click out
def filter_clickout(grp, mode):
    # sessions has clickouts
    # 'clickout item': 2
    has_clickout = (grp['action_type'].values == 2).sum() != 0
    if mode == 'train':
        # last row has reference and it's not nan
        has_ref = ((grp['action_type'].iloc[-1] == 2) &
                   (grp.iloc[-1][['impressions', 'reference', 'prices']].isna().sum() == 0))
    else:
        # test should have the last reference as nan for clickout
        has_ref = ((grp['action_type'].iloc[-1] == 2) &
                   (grp.iloc[-1][['reference']].isna()))
    return has_clickout & has_ref


def clean_sessions(df, data_source='train'):
    df = df.groupby('session_id').apply(clip_last_click).reset_index(drop=True)
    print('filtering out sessions without clickout and reference, or clickout is not valid')
    print(f'length before filtering: {len(df):,}')
    filter_clickout_ = partial(filter_clickout, mode=data_source)
    valid_clicked = df.groupby('session_id').apply(filter_clickout_)
    click_session_ids = valid_clicked[valid_clicked].index
    # filter
    df = df[df.session_id.isin(click_session_ids)].reset_index(drop=True)
    # del valid_clicked, click_session_ids
    gc.collect()
    print(f'{data_source} length after filtering: {len(df):,}')
    return df
