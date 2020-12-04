import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from tests.test_data.test_pipelines.test_loading.test_base_loading import \
    TestLoading

from mmaction.datasets.pipelines import (GenerateLocalizationLabels,
                                         LoadLocalizationFeature)


class TestLocalization(TestLoading):

    def test_load_localization_feature(self):
        target_keys = ['raw_feature']

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_localization_feature = LoadLocalizationFeature(
                'unsupport_ext')

        # test normal cases
        load_localization_feature = LoadLocalizationFeature()
        load_localization_feature_result = load_localization_feature(
            action_result)
        assert self.check_keys_contain(load_localization_feature_result.keys(),
                                       target_keys)
        assert load_localization_feature_result['raw_feature'].shape == (400,
                                                                         5)
        assert repr(load_localization_feature) == (
            f'{load_localization_feature.__class__.__name__}('
            f'raw_feature_ext=.csv)')

    def test_generate_localization_label(self):
        action_result = copy.deepcopy(self.action_results)
        action_result['raw_feature'] = np.random.randn(400, 5)

        # test default setting
        target_keys = ['gt_bbox']
        generate_localization_labels = GenerateLocalizationLabels()
        generate_localization_labels_result = generate_localization_labels(
            action_result)
        assert self.check_keys_contain(
            generate_localization_labels_result.keys(), target_keys)

        assert_array_almost_equal(
            generate_localization_labels_result['gt_bbox'], [[0.375, 0.625]],
            decimal=4)
