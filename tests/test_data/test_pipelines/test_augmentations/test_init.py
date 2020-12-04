import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations


class TestInit(TestAugumentations):

    def test_init_lazy(self):
        from mmaction.datasets.pipelines.augmentations import \
            _init_lazy_if_proper  # noqa: E501
        with pytest.raises(AssertionError):
            # use lazy operation but "lazy" not in results
            result = dict(lazy=dict(), img_shape=[64, 64])
            _init_lazy_if_proper(result, False)

        lazy_keys = [
            'original_shape', 'crop_bbox', 'flip', 'flip_direction',
            'interpolation'
        ]

        # 'img_shape' not in results
        result = dict(imgs=list(np.random.randn(3, 64, 64, 3)))
        _init_lazy_if_proper(result, True)
        assert self.check_keys_contain(result, ['imgs', 'lazy', 'img_shape'])
        assert self.check_keys_contain(result['lazy'], lazy_keys)

        # 'img_shape' in results
        result = dict(img_shape=[64, 64])
        _init_lazy_if_proper(result, True)
        assert self.check_keys_contain(result, ['lazy', 'img_shape'])
        assert self.check_keys_contain(result['lazy'], lazy_keys)

        # do not use lazy operation
        result = dict(img_shape=[64, 64])
        _init_lazy_if_proper(result, False)
        assert self.check_keys_contain(result, ['img_shape'])
        assert 'lazy' not in result
