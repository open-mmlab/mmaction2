import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import RandomRescale


class TestRescale(TestAugumentations):

    def test_random_rescale(self):
        with pytest.raises(AssertionError):
            # scale_range must be a tuple of int
            RandomRescale(scale_range=224)

        with pytest.raises(AssertionError):
            # scale_range must be a tuple of int
            RandomRescale(scale_range=(224.0, 256.0))

        with pytest.raises(AssertionError):
            # scale_range[0] > scale_range[1], which is wrong
            RandomRescale(scale_range=(320, 256))

        with pytest.raises(AssertionError):
            # scale_range[0] <= 0, which is wrong
            RandomRescale(scale_range=(0, 320))

        target_keys = ['imgs', 'short_edge', 'img_shape']
        # There will be a slight difference because of rounding
        eps = 0.01
        imgs = list(np.random.rand(2, 256, 340, 3))
        results = dict(imgs=imgs, img_shape=(256, 340), modality='RGB')

        random_rescale = RandomRescale(scale_range=(300, 400))
        random_rescale_result = random_rescale(results)

        assert self.check_keys_contain(random_rescale_result.keys(),
                                       target_keys)

        h, w = random_rescale_result['img_shape']

        # check rescale
        assert np.abs(h / 256 - w / 340) < eps
        assert 300 / 256 - eps <= h / 256 <= 400 / 256 + eps
        assert repr(random_rescale) == (f'{random_rescale.__class__.__name__}'
                                        f'(scale_range={(300, 400)}, '
                                        'interpolation=bilinear)')
