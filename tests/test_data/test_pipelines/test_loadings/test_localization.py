import copy

import numpy as np
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import GenerateLocalizationLabels
from .base import BaseTestLoading


class TestLocalization(BaseTestLoading):

    def test_generate_localization_label(self):
        action_result = copy.deepcopy(self.action_results)
        action_result['raw_feature'] = np.random.randn(400, 5)

        # test default setting
        target_keys = ['gt_bbox']
        generate_localization_labels = GenerateLocalizationLabels()
        generate_localization_labels_result = generate_localization_labels(
            action_result)
        assert assert_dict_has_keys(generate_localization_labels_result,
                                    target_keys)

        assert_array_almost_equal(
            generate_localization_labels_result['gt_bbox'], [[0.375, 0.625]],
            decimal=4)
