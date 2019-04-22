from clean_session import preprocess_sessions
from session_features import compute_session_fts

from utils import load_data, Fprint


def pipeline(data_source='train', nrows=None):
    fprint = Fprint().fprint
    fprint(f'Start data processing pipeline, first load raw {data_source} data')
    df = load_data(data_source, nrows=nrows)

    fprint('Preprocessing data')
    df = preprocess_sessions(df, data_source=data_source, fprint=fprint)

    fprint('Compute session features')
    session_fts = compute_session_fts(df, data_source=data_source)

    fprint('Done data pipeline')


if __name__ == '__main__':
    data_source = 'train'
    # nrows = 10000
    nrows = None
    pipeline(data_source, nrows=nrows)


