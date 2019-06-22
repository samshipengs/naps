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