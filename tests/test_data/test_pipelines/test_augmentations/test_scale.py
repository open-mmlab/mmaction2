import copy

import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import RandomScale


class TestScale(TestAugumentations):

    def test_random_scale(self):
        scales = ((200, 64), (250, 80))
        with pytest.raises(ValueError):
            RandomScale(scales, 'unsupport')

        with pytest.raises(ValueError):
            random_scale = RandomScale([(800, 256), (1000, 320), (800, 320)])
            random_scale({})

        imgs = list(np.random.rand(2, 340, 256, 3))
        results = dict(imgs=imgs, img_shape=(340, 256))

        results_ = copy.deepcopy(results)
        random_scale_range = RandomScale(scales)
        results_ = random_scale_range(results_)
        assert 200 <= results_['scale'][0] <= 250
        assert 64 <= results_['scale'][1] <= 80

        results_ = copy.deepcopy(results)
        random_scale_value = RandomScale(scales, 'value')
        results_ = random_scale_value(results_)
        assert results_['scale'] in scales

        random_scale_single = RandomScale([(200, 64)])
        results_ = copy.deepcopy(results)
        results_ = random_scale_single(results_)
        assert results_['scale'] == (200, 64)

        assert repr(random_scale_range) == (
            f'{random_scale_range.__class__.__name__}'
            f'(scales={((200, 64), (250, 80))}, '
            'mode=range)')
