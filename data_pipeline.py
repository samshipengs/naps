from clean_session import preprocess_sessions
from session_features import compute_session_fts
from manual_encoding import action_encoding, click_view_encoding, meta_encoding
from utils import load_data, Fprint


def pipeline(data_source='train', nrows=None):
    fprint = Fprint().fprint
    fprint(f'Start data processing pipeline, first load raw {data_source} data')
    df = load_data(data_source, nrows=nrows)

    fprint('Preprocessing data')
    # clip sessions to last clickout and remove sessions with no clickout ( and remove duplicates)
    df = preprocess_sessions(df, data_source=data_source, rd=True, fprint=fprint)

    fprint('Compute session features')
    _ = compute_session_fts(df, data_source=data_source)

    fprint('Getting manual encoding')
    fprint('Action encoding')
    _ = action_encoding()
    fprint('Click view encoding')
    _ = click_view_encoding()
    fprint('Meta encoding')
    _ = meta_encoding()
    fprint('Done manual encodings')




















    fprint('Done data pipeline')


if __name__ == '__main__':
    data_source = 'train'
    # nrows = 10000
    nrows = None
    pipeline(data_source, nrows=nrows)


