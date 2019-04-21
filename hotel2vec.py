import pandas as pd
import os
import gc
from gensim.models import Word2Vec
from utils import ignore_warnings, load_data, Fprint, pshape, get_cpu_count, check_dir


def load_train_test(nrows=None):
    usecols = ['action_type', 'impressions']
    fprint = Fprint().fprint
    fprint('Load train data')
    # train
    train = load_data('train', nrows=nrows, usecols=usecols)
    pshape(train, 'train')
    # test
    fprint('Load test data')
    test = load_data('test', nrows=nrows, usecols=usecols)
    pshape(train, 'test')

    # concat
    return pd.concat([train, test], axis=0, ignore_index=True)


def create_embeddings(nrows=None):
    fprint = Fprint().fprint
    filepath = './cache/hotel_2vec'
    check_dir(filepath)
    filename = os.path.join(filepath, 'model.bin')
    if os.path.isfile(filename):
        fprint(f'Load the existing hotel2vec model from {filename}')
        model = Word2Vec.load(filename)
    else:
        # first load data
        fprint('Load concatenated train and test')
        tt = load_train_test(nrows=nrows)

        fprint("Select only 'clickout item' action type and impressions not na")
        # select the rows that is clickout
        is_clickout = tt['action_type'] == 'clickout item'
        del tt['action_type']
        # and the impressions are not nans
        imp_not_na = tt['impressions'].notna()
        select_mask = is_clickout & imp_not_na
        tt = tt[select_mask].reset_index(drop=True)
        # convert to list of item ids (str)
        tt['impressions'] = tt['impressions'].str.split('|')
        impressions = list(tt['impressions'].values)
        del tt
        gc.collect()

        fprint('Train word2vec embeddings')
        ncpu = get_cpu_count()
        # train model
        model = Word2Vec(impressions, min_count=1, workers=ncpu)
        fprint('Done training, saving model to disk')

        model.save(filename)
        fprint(f'Done saving hotel2vec model to {filename}')
    return model


if __name__ == '__main__':
    _ = create_embeddings()
