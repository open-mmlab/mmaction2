# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import ColorJitter


class TestColor:

    @staticmethod
    def test_color_jitter():
        imgs = list(
            np.random.randint(0, 255, size=(3, 112, 112, 3), dtype=np.uint8))
        results = dict(imgs=imgs)

        color_jitter = ColorJitter()
        assert color_jitter.brightness == (0.5, 1.5)
        assert color_jitter.contrast == (0.5, 1.5)
        assert color_jitter.saturation == (0.5, 1.5)
        assert color_jitter.hue == (-0.1, 0.1)

        color_jitter_results = color_jitter(results)
        target_keys = ['imgs']

        assert assert_dict_has_keys(color_jitter_results, target_keys)
        assert np.shape(color_jitter_results['imgs']) == (3, 112, 112, 3)
        for img in color_jitter_results['imgs']:
            assert np.all(img >= 0)
            assert np.all(img <= 255)

        assert repr(color_jitter) == (f'{color_jitter.__class__.__name__}('
                                      f'brightness={(0.5, 1.5)}, '
                                      f'contrast={(0.5, 1.5)}, '
                                      f'saturation={(0.5, 1.5)}, '
                                      f'hue={-0.1, 0.1})')
