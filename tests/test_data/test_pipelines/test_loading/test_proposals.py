import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import LoadProposals


class TestProposals(TestLoading):

    def test_load_proposals(self):
        target_keys = [
            'bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score',
            'reference_temporal_iou'
        ]

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir,
                                           'unsupport_ext')

        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir, '.csv',
                                           'unsupport_ext')

        # test normal cases
        load_proposals = LoadProposals(5, self.proposals_dir,
                                       self.bsp_feature_dir)
        load_proposals_result = load_proposals(action_result)
        assert self.check_keys_contain(load_proposals_result.keys(),
                                       target_keys)
        assert (load_proposals_result['bsp_feature'].shape[0] == 5)
        assert load_proposals_result['tmin'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin'], np.arange(0.1, 0.6, 0.1), decimal=4)
        assert load_proposals_result['tmax'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax'], np.arange(0.2, 0.7, 0.1), decimal=4)
        assert load_proposals_result['tmin_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin_score'],
            np.arange(0.95, 0.90, -0.01),
            decimal=4)
        assert load_proposals_result['tmax_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax_score'],
            np.arange(0.96, 0.91, -0.01),
            decimal=4)
        assert load_proposals_result['reference_temporal_iou'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['reference_temporal_iou'],
            np.arange(0.85, 0.80, -0.01),
            decimal=4)
        assert repr(load_proposals) == (
            f'{load_proposals.__class__.__name__}('
            f'top_k={5}, '
            f'pgm_proposals_dir={self.proposals_dir}, '
            f'pgm_features_dir={self.bsp_feature_dir}, '
            f'proposal_ext=.csv, '
            f'feature_ext=.npy)')
