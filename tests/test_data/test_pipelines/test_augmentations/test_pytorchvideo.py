# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys, digit_version

try:
    import torch

    from mmaction.datasets.pipelines import PytorchVideoTrans
    pytorchvideo_ok = False
    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        pytorchvideo_ok = True
except (ImportError, ModuleNotFoundError):
    pytorchvideo_ok = False


@pytest.mark.skipif(not pytorchvideo_ok, reason='torch >= 1.8.0 is required')
class TestPytorchVideoTrans:

    @staticmethod
    def test_pytorchvideo_trans():
        with pytest.raises(AssertionError):
            # transforms not supported in pytorchvideo
            PytorchVideoTrans(type='BlaBla')

        with pytest.raises(AssertionError):
            # This trans exists in pytorchvideo but not supported in MMAction2
            PytorchVideoTrans(type='MixUp')

        target_keys = ['imgs']

        imgs = list(np.random.randint(0, 256, (4, 32, 32, 3)).astype(np.uint8))
        results = dict(imgs=imgs)

        # test AugMix
        augmix = PytorchVideoTrans(type='AugMix')
        results = augmix(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (32, 32, 3) for img in results['imgs']))

        # test RandAugment
        rand_augment = PytorchVideoTrans(type='RandAugment')
        results = rand_augment(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (32, 32, 3) for img in results['imgs']))

        # test RandomResizedCrop
        random_resized_crop = PytorchVideoTrans(
            type='RandomResizedCrop',
            target_height=16,
            target_width=16,
            scale=(0.1, 1.),
            aspect_ratio=(0.8, 1.2))
        results = random_resized_crop(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (16, 16, 3) for img in results['imgs']))

        # test ShortSideScale
        short_side_scale = PytorchVideoTrans(type='ShortSideScale', size=24)
        results = short_side_scale(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (24, 24, 3) for img in results['imgs']))

        # test ShortSideScale
        random_short_side_scale = PytorchVideoTrans(
            type='RandomShortSideScale', min_size=24, max_size=36)
        results = random_short_side_scale(results)
        target_shape = results['imgs'][0].shape
        assert 36 >= target_shape[0] >= 24
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == target_shape for img in results['imgs']))
