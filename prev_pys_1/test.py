import unittest
import pandas as pd
from utils import load_data
from hotel2vec import load_train_test


class TestHotelVec(unittest.TestCase):
    def test_size(self):
        train = load_data('train', usecols=['session_id'])
        test = load_data('test', usecols=['session_id'])
        tt = load_train_test()
        self.assertEqual(tt.shape[0], train.shape[0]+test.shape[0],
                         msg='Size of train test concat should be the sum of each size')


class TestSessionFeatures(unittest.TestCase):
    """
    session_aggs = {'timestamp': [span, mean_dwell_time, var_dwell_time, median_dwell_time, dwell_time_before_last],
                    'step': ['max'],prev_pys_1
                    'action_type': ['nunique', n_clickouts, click_rel_pos_avg, second_last],
                    'reference': [second_last],
                    'city': ['nunique', get_last],
                    'platform': [get_last],
                    'device': [get_last],
                    'n_imps': [get_last],
                    'n_filters': [get_last]}
    """
    def __init__(self, *args, **kwargs):
        super(TestSessionFeatures, self).__init__(*args, **kwargs)
        self.train_fts = pd.read_hdf('./cache/', 'train')
        self.test_fts = pd.read_hdf('./cache/', 'test')

    def test_timestamp_span(self):
        # the span should not be negative
        self.assertTrue((self.train_fts['timestamp_span'].dropna() > 0).all(),
                        'timestamp span should be greater than 0')
        # the max timespan in train is 404403
        self.assertTrue((self.train_fts['timestamp_span'].dropna() > 404403).all(),
                        'timestamp span should be greater less than 404403')
        self.assertTrue((self.train_fts['mean_dwell_time'].dropna() > 0).all(),
                        'timestamp mean_dwell_time should be greater than 0')
        self.assertTrue((self.train_fts['mean_dwell_time'].dropna() > 0).all(),
                        'timestamp mean_dwell_time should be greater than 0')
        self.assertTrue((self.train_fts['median_dwell_time'].dropna() > 0).all(),
                        'timestamp median_dwell_time should be greater than 0')
        self.assertTrue((self.train_fts['dwell_time_before_last'].dropna() > 0).all(),
                        'timestamp dwell_time_before_last should be greater than 0')

        # the span should not be negative
        self.assertTrue((self.test_fts['timestamp_span'].dropna() > 0).all(),
                        'timestamp span should be greater than 0')
        # the max timespan in train is 404403
        self.assertTrue((self.test_fts['timestamp_span'].dropna() > 404403).all(),
                        'timestamp span should be less than 403303')
        self.assertTrue((self.test_fts['mean_dwell_time'].dropna() > 0).all(),
                        'timestamp mean_dwell_time should be greater than 0')
        self.assertTrue((self.test_fts['var_dwell_time'].dropna() > 0).all(),
                        'timestamp var_dwell_time should be greater than 0')
        self.assertTrue((self.test_fts['median_dwell_time'].dropna() > 0).all(),
                        'timestamp median_dwell_time should be greater than 0')
        self.assertTrue((self.test_fts['dwell_time_before_last'].dropna() > 0).all(),
                        'timestamp dwell_time_before_last should be greater than 0')

    def test_step(self):
        self.assertTrue((self.train_fts['step_max'] > 0).all(), 'step should be positive number')
        self.assertTrue((self.test_fts['step_max'] > 0).all(), 'step should be positive number')

    def test_action_type(self):
        pass

    def test_current_filters(self):
        pass


if __name__ == '__main__':
    unittest.main()
