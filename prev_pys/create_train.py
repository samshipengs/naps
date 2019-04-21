import time
import gc

from reduce_data import reduce_data
from clean_session import clean_sessions
from create_meta_fts import create_meta_fts
from create_session_fts import create_session_fts
from create_item_fts import generate_session_item_pairs
from utils import ignore_warnings


ignore_warnings()


def create_train(data_source, nrows=None):
    t1 = time.time()
    fprint = lambda msg: print(f"{msg:<40} {'=' * 20} time elapsed = {(time.time() - t1) / 60:.2f} mins")

    train, meta, mapper_dict = reduce_data(nrows)
    fprint('Done initial reducing')
    train = clean_sessions(train, data_source=data_source)
    fprint('Done cleaning')
    meta_fts = create_meta_fts(meta, train, mapper_dict)
    del meta
    gc.collect()
    fprint('Done meta features')
    session_fts = create_session_fts(train)
    fprint('Done session features')
    items_fts = generate_session_item_pairs(data_source, train, meta_fts, mapper_dict)
    del train, meta_fts
    gc.collect()
    fprint('Done item features')

    # join
    items_fts.reset_index(level='session_id', inplace=True)
    items_fts.set_index('session_id', inplace=True)
    train_df = items_fts.join(session_fts)
    fprint('Done all')
    train_df.to_hdf('./data/train.h5', key='train')
    fprint(f'Done saving, train shape = {train_df.shape}')


if __name__ == '__main__':
    nrows = None
    # nrows = 100000
    create_train('train', nrows=nrows)
