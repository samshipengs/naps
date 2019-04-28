import os
import pandas as pd
import numpy as np
from utils import load_data, check_dir, get_logger


logger = get_logger('manual_encoding')


def action_encoding(nrows=None, per_session=True, save=True, recompute=False):
    filepath = './cache'
    filename = os.path.join(filepath, 'action_encodings.csv')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from exsiting file: {filename}')
        encoding = pd.read_csv(filename)
    else:
        # only load reference and action_type
        ref_act = load_data('train', nrows=nrows, usecols=['session_id', 'reference', 'action_type'])
        # remove references that were not integers (we could include them but these are
        # normally associated with destination search related actions)
        ref_act = ref_act[~ref_act.reference.str.contains('[a-zA-Z]')].reset_index(drop=True)
        ref_act['reference'] = ref_act['reference'].astype(int)
        # get all the unique actions
        unique_actions = ref_act['action_type'].unique()
        logger.info(f'The unique actions to encode are:\n{unique_actions}')
        # encode them in one-hot
        action2natural = {v: k for k, v in enumerate(unique_actions)}
        action_names = list(action2natural.keys())
        actions = ref_act['action_type'].map(action2natural).values
        # create the ohe df
        actions_ohe = pd.DataFrame(np.eye(len(unique_actions), dtype=int)[actions], columns=action_names)

        # merge it back
        ref_act = pd.concat([ref_act, actions_ohe], axis=1)

        # if the counts of an action is summed within a session
        if per_session:
            ref_act = ref_act.groupby(['session_id', 'reference', 'action_type']).sum().reset_index()
            del ref_act['session_id']
        del ref_act['action_type']

        # use smoothed encoding
        M = [5] * len(action_names)
        logger.info(f'Smooth with means weights: {M}')
        s = []
        for k, c in enumerate(action_names):
            logger.info(f'smooth encoding: {c}')
            # Compute the global mean
            mu = ref_act[c].mean()

            # Compute the number of values and the mean of each group
            agg = ref_act.groupby('reference')[c].agg(['count', 'mean'])
            counts = agg['count']
            mus = agg['mean']
            # Compute the "smoothed" means
            # smoothed = (n*mu + m*Mu)/(n+m)
            smoothed = (counts * mus + M[k] * mu) / (counts + M[k])
            # ref_act[c] = ref_act['reference'].map(smoothed)
            s.append(smoothed.reset_index(name=c))
        del ref_act
        dfs = [df.set_index('reference') for df in s]
        encoding = pd.concat(dfs, axis=1).reset_index()
        # ref_act.drop_duplicates(inplace=True)
        # ref_act.reset_index(drop=True, inplace=True)
        check_dir(filepath)
        if save:
            encoding.to_csv(filename, index=False)
    return encoding


def click_view_encoding(m=5, nrows=None, recompute=False):
    """
    encode click and view
    :param m: smoothing factor
    :param nrows: load numebr of rows data
    :param recompute:
    :return:
    """
    filepath = './cache'
    filename = os.path.join(filepath, 'clickview_encodings.csv')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from exsiting file: {filename}')
        encoding = pd.read_csv(filename)
    else:
        # only load reference and action_type
        ref_imp = load_data('train', nrows=nrows, usecols=['action_type', 'reference', 'impressions'])
        ref_imp = ref_imp.loc[ref_imp.action_type == 'clickout item'].reset_index(drop=True)
        del ref_imp['action_type']
        ref_imp = ref_imp[~ref_imp.reference.str.contains('[a-zA-Z]')].reset_index(drop=True)

        # create list of impressions
        ref_imp['impressions'] = ref_imp['impressions'].str.split('|')
        # remove the clicked id from impressions
        ref_imp['impressions'] = ref_imp.apply(lambda row: list(set(row.impressions) - set([row.reference])), axis=1)

        # create 0 for impressions (viewed) and 1 for clicked
        imps = ref_imp.impressions.values
        imps = [j for i in imps for j in i]
        click_imp = pd.concat([pd.Series([0] * len(imps), index=imps),
                               pd.Series([1] * len(ref_imp), index=ref_imp.reference.values)])
        click_imp.index.name = 'item_id'
        click_imp = pd.DataFrame(click_imp, columns=['clicked']).reset_index()

        # smoothed encoding
        mu = click_imp['clicked'].mean()
        agg = click_imp.groupby('item_id')['clicked'].agg(['count', 'mean'])
        count = agg['count']
        mus = agg['mean']
        smoothed = (count * mus + m * mu) / (count + m)
        # click_imp['clicked'] = click_imp['item_id'].map(smoothed)
        encoding = smoothed.reset_index(name='clicked')
        encoding['item_id'] = encoding['item_id'].astype(int)

        # save
        check_dir(filepath)
        encoding.to_csv(filename, index=False)
    return encoding


def meta_encoding(recompute=False):
    filepath = './cache'
    filename = os.path.join(filepath, 'meta_encodings.csv')
    if os.path.isfile(filename) and not recompute:
        logger.info(f'Load from exsiting file: {filename}')
        encoding = pd.read_csv(filename)
    else:
        meta = load_data('item_metadata')
        # get list of properties
        meta['properties'] = meta.properties.str.lower()
        meta['properties'] = meta['properties'].str.split('|')

        # create mapping
        properties = meta.properties.values
        properties = list(set([j for i in properties for j in i]))
        property_mapping = {v: k for k, v in enumerate(properties)}
        property_names = list(property_mapping.keys())
        meta['properties'] = meta.properties.apply(lambda l: [property_mapping[i] for i in l])
        # create zeros for encoding first
        zeros = np.zeros((len(meta), len(property_mapping.keys())), dtype=int)
        # then assign
        ps = meta.properties
        for i in range(meta.shape[0]):
            zeros[i, ps[i]] = 1

        encoding = pd.DataFrame(zeros, columns=property_names, index=meta.item_id).reset_index()
        encoding['item_id'] = encoding['item_id'].astype(int)
        # save
        check_dir(filepath)
        encoding.to_csv(filename, index=False)

    return encoding


if __name__ == '__main__':
    logger.info('Action encoding')
    _ = action_encoding()
    logger.info('Click view encoding')
    _ = click_view_encoding()
    logger.info('Meta encoding')
    _ = meta_encoding()
    logger.info('Done manual encodings')


