import os
import gc
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from utils import ignore_warnings, load_data, Fprint, pshape, get_cpu_count, check_dir
from clean_session import remove_duplicates


def load_train_test(nrows=None, rd=True):
    usecols = ['action_type', 'impressions']
    fprint = Fprint().fprint
    fprint('Load train data')
    # train
    train = load_data('train', nrows=nrows)
    pshape(train, 'train')
    if rd:
        fprint('Removing duplicates when training word2vec')
        train = remove_duplicates(train)
    train = train[usecols]
    # test
    fprint('Load test data')
    test = load_data('test', nrows=nrows)#, usecols=usecols)
    pshape(train, 'test')
    test = remove_duplicates(test)
    test = test[usecols]
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


def hotel2vec(model=None):
    fprint = Fprint().fprint
    filepath = './cache/hotel_2vec'
    check_dir(filepath)
    filename = os.path.join(filepath, 'embeddings.csv')
    if os.path.isfile(filename):
        fprint(f'Load the existing hotel2vec model from {filename}')
        embeddings = pd.read_csv(filename)
    else:
        assert model is not None, 'hotel2vec model is required'
        train = load_data('train', usecols=['impressions'])
        train.dropna(inplace=True)
        train.drop_duplicates(inplace=True)
        train['impressions'] = train['impressions'].str.split('|')
        impressions = train['impressions'].values
        impressions = list(set([j for i in impressions for j in i]))
        embeddings = np.array([model.wv[i] for i in impressions])
        cols = [f'hv{i}' for i in range(embeddings.shape[1])]
        embeddings = pd.DataFrame(embeddings, columns=cols, index=impressions)
        embeddings.index.name = 'item_id'
        embeddings.reset_index(inplace=True)
        embeddings['item_id'] = embeddings['item_id'].astype(int)
        embeddings.to_csv(filename)
    return embeddings




if __name__ == '__main__':
    model = create_embeddings()
    _ = hotel2vec(model)
