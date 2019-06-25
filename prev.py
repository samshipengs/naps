# def meta_encoding(recompute=False):
#     """
#     Get encoding, i.e. the properties from meta csv
#     :param recompute:
#     :return:
#         meta_mapping: dict, mapping {123: [1, 0, 0, 1, 0, ...], ...}
#         n_properties: number of unique properties
#     """
#     filepath = Filepath.cache_path
#     filename = os.path.join(filepath, 'meta_mapping.npy')
#     if os.path.isfile(filename) and not recompute:
#         logger.info(f'Load from existing file: {filename}')
#         meta_mapping = np.load(filename).item()
#         n_properties = len(meta_mapping[list(meta_mapping.keys())[0]])
#     else:
#         logger.info('Load meta data')
#         meta_df = load_data('item_metadata')
#
#         logger.info('Lower and split properties str to list')
#         meta_df['properties'] = meta_df['properties'].str.lower().str.split('|')
#
#         logger.info('Get all unique properties')
#         unique_properties = list(set(np.concatenate(meta_df['properties'].values)))
#         property2natural = {v: k for k, v in enumerate(unique_properties)}
#         n_properties = len(unique_properties)
#         logger.info(f'Total number of unique meta properties: {n_properties}')
#
#         logger.info('Convert the properties to ohe and superpose for each item_id')
#         meta_df['properties'] = meta_df['properties'].apply(lambda ps: [property2natural[p] for p in ps])
#         meta_df['properties'] = meta_df['properties'].apply(lambda ps: np.sum(np.eye(n_properties, dtype=np.int16)[ps],
#                                                                               axis=0))
#         logger.info('Create mappings')
#         meta_mapping = dict(meta_df[['item_id', 'properties']].values)
#         # add a mapping (all zeros) for the padded impression ids
#         # meta_mapping[0] = np.zeros(n_properties, dtype=int)
#         logger.info('Saving meta mapping')
#         np.save(filename, meta_mapping)
#     return meta_mapping, n_properties


def find_relative_ref_loc(row):
    last_action_type, last_ref = row['last_action_type'], row['last_reference']
    if pd.isna(last_ref) or type(row['impressions']) != list:
        # first row
        return [np.nan, np.nan]
    else:
        imps = list(row['impressions'])
        if last_ref in imps:
            if last_action_type == 0:
                return [(imps.index(last_ref)+1)/25, np.nan]
            else:
                return [np.nan, (imps.index(last_ref)+1)/25]
        else:
            return [np.nan, np.nan]


# class LoggingCallback(Callback):
#     """Callback that logs message at end of epoch.
#     """
#     def __init__(self, print_fcn=print):
#         Callback.__init__(self)
#         self.print_fcn = print_fcn
#
#     def on_epoch_end(self, epoch, logs={}):
#         msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in logs.items()))
#         self.print_fcn(msg)


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


# def iterate_minibatches(numerics, prices, clicks, targets, batch_size, shuffle=True):
#     # default we will shuffle
#     indices = np.arange(len(targets))
#     while True:
#         if shuffle:
#             np.random.shuffle(indices)
#
#         remainder = len(targets) % batch_size
#         for start_idx in range(0, len(targets), batch_size):
#             if remainder != 0 and start_idx + batch_size >= len(targets):
#                 excerpt = indices[len(targets) - batch_size:len(targets)]
#             else:
#                 excerpt = indices[start_idx:start_idx + batch_size]
#
#             numerics_batch = numerics[excerpt]
#             prices_batch = prices[excerpt]
#             clicks_batch = clicks[excerpt]
#             targets_batch = targets[excerpt]
#
#             yield ([numerics_batch, prices_batch, clicks_batch], targets_batch)

def compute_session_func(grp):
    """
    Main working function to compute features or shaping data
    :param grp: dataframe associated with a group key (session_id) from groupby
    :return: dataframe with feature columns
    """
    df = grp.copy()
    # number of records in session
    df['session_size'] = list(range(1, len(df) + 1))

    # session_time duration (subtract the min)
    df['session_duration'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    # but if it is the first row, it is always nan
    if len(df) == 0:
        df['session_duration'] = np.nan

    # get consecutive time difference
    df['last_duration'] = df['timestamp'].diff().dt.total_seconds()
    # maybe we don't fill na with 0 as it could be implying a very short time interval
    df.drop('timestamp', axis=1, inplace=True)

    # instead of ohe, just use it as categorical input
    # shift down
    df['last_action_type'] = df['action_type'].shift(1)
    df['fs'] = df['fs'].shift(1)
    df['sort_order'] = df['sort_order'].shift(1)

    # in case there is just one row, use list comprehension instead of np concat
    impressions = df['impressions'].dropna().values
    unique_items = list(set([j for i in impressions for j in i] + list(df['reference'].unique())))
    temp_mapping = {v: k for k, v in enumerate(unique_items)}
    df['reference_natural'] = df['reference'].map(temp_mapping)
    click_out_mask = df['action_type'] == 0  # 0 is the hard-coded encoding for clickout item
    df.drop('action_type', axis=1, inplace=True)
    other_mask = ~click_out_mask
    click_cols = [f'click_{i}' for i in range(len(unique_items))]
    interact_cols = [f'interact_{i}' for i in range(len(unique_items))]

    # create click out binary encoded dataframe
    click_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[click_out_mask]['reference_natural'].values],
                            columns=click_cols, index=df[click_out_mask].index)
    # create everything else except clickout
    interact_df = pd.DataFrame(np.eye(len(unique_items), dtype=int)[df[other_mask]['reference_natural'].values],
                               columns=interact_cols, index=df[other_mask].index)
    df = pd.concat([df, click_df, interact_df], axis=1)
    df.drop('reference_natural', axis=1, inplace=True)

    # first get all previous clicks, and others
    df_temp = df.copy()
    df_temp.fillna(0, inplace=True)  # we need to fill na otherwise the cumsum would not compute on row with nan value
    prev_click_df = df_temp[click_cols].shift(1).cumsum()
    prev_interact_df = df_temp[interact_cols].shift(1).cumsum()

    # remove the original click and interact cols, swap on the last and prev clicks and interaction cols
    df.drop(click_cols + interact_cols, axis=1, inplace=True)
    # concat all
    df = pd.concat([df, prev_click_df, prev_interact_df], axis=1)

    # add impression relative location
    df['last_reference'] = df['reference'].shift(1)
    df.loc[click_out_mask, 'last_reference_relative_loc'] = df.loc[click_out_mask].apply(find_relative_ref_loc, axis=1)
    df.drop('last_reference', axis=1, inplace=True)

    # only need click-out rows now
    df = df[click_out_mask].reset_index(drop=True)

    # now select only the ones that is needed for each row
    def match(row, cols):
        impressions_natural = [temp_mapping[imp] for imp in row['impressions']]
        return row[cols].values[impressions_natural]

    iter_cols = {'prev_click': click_cols, 'prev_interact': interact_cols}
    for k, v in iter_cols.items():
        func = partial(match, cols=v)
        df[k] = df.apply(func, axis=1)
        df.drop(v, axis=1, inplace=True)

    return df