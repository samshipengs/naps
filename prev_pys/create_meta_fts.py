import pandas as pd
import numpy as np
import os


def create_rating_colum(meta):
    # create rating columns
    ratings = ['good rating', 'satisfactory rating', 'excellent rating']
    for r in ratings:
        meta[r.replace(' ', '_')] = meta['properties'].str.findall(f'(\||\b){r}').str.len()


def action_encoding(df, mapper_dict):
    # now group on reference ids
    action_grp = df.groupby('reference')['action_type']
    # get value counts for each action type
    action_ctn = action_grp.value_counts()
    action_ctn_df = action_ctn.reset_index(name='ctn')

    # list of all unique action type
    # actions = list(np.sort(df.action_type.unique()))
    action_mapper = mapper_dict['action_type']
    # actions_id = action_mapper.values()
    actions_name = list(action_mapper.keys())

    # create ohe
    ohe = pd.DataFrame(np.eye(len(actions_name), dtype=int)[action_ctn_df.action_type.values],
                       columns=actions_name)
    ohe = ohe.mul(action_ctn_df['ctn'], axis=0)
    action_ctn_df = pd.concat([action_ctn_df, ohe], axis=1)

    action_encoding = action_ctn_df.groupby('reference')[actions_name].sum()
    # also add normalized percentage over count of each actions over total
    normalized = action_encoding.div(action_encoding.sum(axis=1)+1, axis=0) # +1 for smoothing avoiding leakage
    action_encoding = action_encoding.join(normalized, lsuffix='_ctn', rsuffix='_per')
    # set the popularity (i.e the number of clickout) counts to rank, avoid leakage
    # but even if we convert them to rank it still leaks (e.g. 0 clicks out will always stay behind a rank threshold)
    # so for now we drop it (try to use embeddings)
    del action_encoding['clickout item_ctn']
    return action_encoding.reset_index()


def create_meta_fts(meta, df, mapper_dict, recompute=False):
    meta_file = f'./data/meta_fts.csv'
    if os.path.isfile(meta_file) and not recompute:
        print(f'{meta_file} exists, reload')
        meta = pd.read_csv(meta_file)
        # # convert 'list' to list
        # meta.loc[:, 'ps'] = meta.loc[:, 'ps'].apply(literal_eval)
        # meta['ps'] = meta.ps.apply(lambda x: [int(i) for i in x])
        meta = meta.set_index('item_id')
    else:
        # add ratings
        create_rating_colum(meta)

        meta['ps'] = meta['properties_int'].str.split('|')
        #         meta['ps'] = meta['ps'].apply(lambda x: [int(i) for i in x])
        # numer of properties
        meta['nprop'] = meta['ps'].str.len()
        # star ratings
        meta['star'] = meta['properties'].str.extract('[\|](\d) star')
        meta['star'] = meta['star'].astype(float)
        del meta['properties']
        meta.rename(columns={'properties_int': 'properties'}, inplace=True)

        # action encodings
        action_encodings = action_encoding(df, mapper_dict)
        meta = pd.merge(meta, action_encodings, left_on='item_id', right_on='reference')
        # choose columns
        act_cols = [c for c in action_encodings.columns if c != 'reference']

        use_cols = ['item_id', 'nprop', 'star', 'good_rating', 'satisfactory_rating',
                    'excellent_rating', 'properties']
        use_cols += act_cols
        meta = meta[use_cols].set_index('item_id')
        meta.to_csv(meta_file)
    return meta