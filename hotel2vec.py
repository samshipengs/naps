import os
import gc
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from utils import ignore_warnings, load_data, get_logger, pshape, get_cpu_count, check_dir
from clean_session import remove_duplicates


logger = get_logger('hotel2vec')


def load_train_test(nrows=None, rd=True):
    usecols = ['action_type', 'impressions']
    logger.info('Load train data')
    # train
    train = load_data('train', nrows=nrows)
    pshape(train, 'train')
    if rd:
        logger.info('Removing duplicates when training word2vec')
        train = remove_duplicates(train)
    train = train[usecols]
    # test
    logger.info('Load test data')
    test = load_data('test', nrows=nrows)#, usecols=usecols)
    pshape(train, 'test')
    test = remove_duplicates(test)
    test = test[usecols]
    # concat
    return pd.concat([train, test], axis=0, ignore_index=True)


def create_embeddings(nrows=None):
    filepath = './cache/hotel_2vec'
    check_dir(filepath)
    filename = os.path.join(filepath, 'model.bin')
    if os.path.isfile(filename):
        logger.info(f'Load the existing hotel2vec model from {filename}')
        model = Word2Vec.load(filename)
    else:
        # first load data
        logger.info('Load concatenated train and test')
        tt = load_train_test(nrows=nrows)

        logger.info("Select only 'clickout item' action type and impressions not na")
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

        logger.info('Train word2vec embeddings')
        ncpu = get_cpu_count()
        # train model
        model = Word2Vec(impressions, min_count=1, workers=ncpu)
        logger.info('Done training, saving model to disk')

        model.save(filename)
        logger.info(f'Done saving hotel2vec model to {filename}')
    return model


def hotel2vec():
    filepath = './cache/hotel_2vec'
    check_dir(filepath)
    filename = os.path.join(filepath, 'embeddings.csv')
    if os.path.isfile(filename):
        logger.info(f'Load the existing hotel2vec model from {filename}')
        embeddings = pd.read_csv(filename)
    else:
        model = create_embeddings()
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
    _ = hotel2vec()
