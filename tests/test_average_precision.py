import os.path as osp
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mmaction.core import (frame_mean_ap, frame_mean_ap_error, pr_to_ap,
                           video_mean_ap)


def test_pr_to_ap():
    precision_recall = np.array([[1., 0.], [1., 0.028], [0.5, 0.028],
                                 [0.027, 0.886]])
    average_precision = pr_to_ap(precision_recall)
    assert average_precision == 0.254083


class TestTubeMetrics:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), 'data', 'test_tubes')
        cls.tube_dir = osp.join(cls.data_prefix, 'tubes')
        cls.gt_tubes_file = osp.join(cls.data_prefix, 'gt_tubes.pkl')
        cls.tube_sample_file = osp.join(cls.data_prefix, 'tube_sample.pkl')
        cls.det_file = osp.join(cls.data_prefix, 'all_dets.npy')
        cls.labels = ('Basketball', 'BasketballDunk')
        cls.videos = ('Basketball/v_Basketball_g01_c01',
                      'BasketballDunk/v_BasketballDunk_g01_c01')
        cls.gt_tubes = pickle.load(open(cls.gt_tubes_file, 'rb'))
        cls.det_results = np.load(cls.det_file)

    @pytest.mark.parametrize('threshold,v_ap', [(0.5, 0), (0.3, 0.5),
                                                (0.25, 0.5), (0.2, 1)])
    def test_video_ap(self, threshold, v_ap):
        with pytest.raises(FileNotFoundError):
            video_mean_ap(self.labels, self.videos, self.gt_tubes, 'not_found')

        result = video_mean_ap(self.labels, self.videos, self.gt_tubes,
                               self.tube_dir, threshold)
        assert v_ap == result

    @pytest.mark.parametrize('threshold,f_ap', [(0.7, 0.20534295),
                                                (0.75, 0.1337894),
                                                (0.8, 0.044360217),
                                                (0.85, 0.0)])
    def test_frame_ap(self, threshold, f_ap):
        result = frame_mean_ap(self.det_results, self.labels, self.videos,
                               self.gt_tubes, threshold)
        assert_array_almost_equal(result, f_ap)

    @pytest.mark.parametrize('threshold,ap,le,ce,te,oe,miss',
                             [(0.7, (41.068584, 0.), (0., 0.), (0., 0.),
                               (4.385963, 0.), (0., 0.), (54.545452, 100.)),
                              (0.75, (26.757877, 0.), (8.14467, 0.), (0., 0.),
                               (3.7338152, 0.), (0., 0.), (61.363636, 100.)),
                              (0.8, (8.872042, 0.), (11.507832, 0.), (0., 0.),
                               (2.3474007, 0.), (0., 0.), (77.27273, 100.))])
    def test_frame_ap_error(self, threshold, ap, le, ce, te, oe, miss):
        gts = [ap, le, ce, te, oe, miss]
        results = frame_mean_ap_error(self.det_results, self.labels,
                                      self.videos, self.gt_tubes, threshold)
        for result, gt in zip(results.values(), gts):
            gt = np.array(gt, dtype=np.float32)
            assert_array_almost_equal(result, gt)
