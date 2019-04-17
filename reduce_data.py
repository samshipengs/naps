import numpy as np
import time
import re
from functools import partial

from utils import load_data


def reduce_data(nrows=None):
    train = load_data('train', nrows=nrows, verbose=True)
    meta = load_data('item_metadata', verbose=True)

    cols_lower = ['action_type', 'reference', 'platform', 'city', 'device', 'current_filters']
    for c in cols_lower:
        print('lowering ', c)
        train[c] = train[c].str.lower()

    meta['properties'] = meta['properties'].str.lower()

    create_mapper = lambda unique_values: {v: k for k, v in enumerate(unique_values)}
    def convert(df, col, mapper):
        df.loc[df[col].notna(), col] = df.loc[df[col].notna()][col].map(mapper)

    def replace(str_list, replace_dict):
        return '|'.join([replace_dict[s] for s in str_list])

    # reduce all data types to int and cache the mapping used
    t_int = time.time()
    fprint = lambda msg: print(f"{msg:<40} {'=' * 20} time elapsed = {(time.time() - t_int) / 60:.2f} mins")
    # ===========================================================================================
    # 0) all session_ids and user_ids
    unique_user_ids = train.user_id.unique()
    user_id_mapper = create_mapper(unique_user_ids)
    convert(train, 'user_id', user_id_mapper)
    fprint('done user_id')

    unique_session_ids = train.session_id.unique()
    session_id_mapper = create_mapper(unique_session_ids)
    convert(train, 'session_id', session_id_mapper)
    fprint('done session_id')
    # ===========================================================================================
    # 1) get timestamp range and subtract the min
    min_ts = train['timestamp'].min()
    train['timestamp'] -= min_ts
    fprint('done timestamp')
    # ===========================================================================================
    # 2) action_type
    unique_action_types = train.action_type.dropna().unique()
    action_mapper = create_mapper(unique_action_types)
    convert(train, 'action_type', action_mapper)
    fprint('done action_type')
    # ===========================================================================================
    # # 3) all item_ids and reference
    impression_lists = train[train.impressions.notna()].impressions.str.split('|')
    unique_impressions = list(set([j for i in impression_lists for j in i]))
    meta_ids = list(meta['item_id'].astype(str).unique())
    complete_item_ids = (list(train[train.reference.notna()].reference.unique())
                         + unique_impressions
                         + meta_ids)
    reference_item_id_mapper = create_mapper(complete_item_ids)
    reference_item_id_mapper_str = {k: str(v) for k, v in reference_item_id_mapper.items()}
    reference_item_id_mapper_int = {int(k): v for k, v in reference_item_id_mapper.items()
                                    if not re.search('[a-zA-Z]', k)}
    convert(train, 'reference', reference_item_id_mapper)
    fprint('done reference')

    replace_impression = partial(replace, replace_dict=reference_item_id_mapper_str)
    train['imps'] = train['impressions'].str.split('|')
    train.loc[train['impressions'].notna(), 'imps'] = train[train['impressions'].notna()]['imps'].apply(
        replace_impression)
    del train['impressions']
    fprint('done impressions item_id for train')
    convert(meta, 'item_id', reference_item_id_mapper_int)
    fprint('done item_id for meta')

    # ===========================================================================================
    # 4) all platform
    unique_platform = train.platform.dropna().unique()
    platform_mapper = create_mapper(unique_platform)
    convert(train, 'platform', platform_mapper)
    fprint('done platform')
    # ===========================================================================================

    # 4) all cities
    unique_cities = train.city.dropna().unique()
    city_mapper = create_mapper(unique_cities)
    convert(train, 'city', city_mapper)
    fprint('done city')
    # ===========================================================================================

    # 5) all device
    unique_device = train.device.dropna().unique()
    device_mapper = create_mapper(unique_device)
    convert(train, 'device', device_mapper)
    fprint('done device')
    # ===========================================================================================

    # 6) filters/properties
    # unique filters from filters
    filter_lists = train[train.current_filters.notna()].current_filters.str.split('|')
    unique_filters = list(set([j for i in filter_lists for j in i]))
    # unique properties from meta
    properties_lists = meta.properties.str.split('|')
    unique_properties = list(set([j for i in properties_lists for j in i]))
    all_properties = list(set(unique_filters + unique_properties))
    properties_mapper = create_mapper(all_properties)
    properties_mapper_str = {k: str(v) for k, v in properties_mapper.items()}

    replace_properties = partial(replace, replace_dict=properties_mapper_str)
    train['cf'] = train['current_filters'].str.split('|')
    train.loc[train['cf'].notna(), 'cf'] = train[train['cf'].notna()]['cf'].apply(replace_properties)
    del train['current_filters']
    fprint('done cf')

    meta['ps_list'] = meta.properties.str.split('|')
    meta['properties_int'] = meta.ps_list.apply(replace_properties)
    del meta['ps_list']  # , meta['properties']
    fprint('done meta properties')

    # rename
    train.rename(columns={'imps': 'impressions', 'cf': 'current_filters'}, inplace=True)

    # save all the mappings
    mapper_dict = {}
    mapper_dict['user_id'] = user_id_mapper
    mapper_dict['session_id'] = session_id_mapper
    mapper_dict['action_type'] = action_mapper
    mapper_dict['reference_item'] = reference_item_id_mapper
    mapper_dict['user_id'] = user_id_mapper
    mapper_dict['platfrom'] = platform_mapper
    mapper_dict['city'] = city_mapper
    mapper_dict['device'] = device_mapper
    mapper_dict['properties'] = properties_mapper


    train.to_hdf('./data/train_reduced.h5', key='train')
    meta.to_hdf('./data/meta_reduced.h5', key='meta')
    np.save('./data/mapper.npy', mapper_dict)

    return train, meta, mapper_dict


