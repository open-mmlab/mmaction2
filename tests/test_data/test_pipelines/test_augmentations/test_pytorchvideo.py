import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import PytorchVideoTrans


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
        imgs = list(np.random.rand(4, 32, 32, 3).astype(np.float32))
        results = dict(imgs=imgs)

        # test AugMix
        augmix = PytorchVideoTrans(type='AugMix')
        results = augmix(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (32, 32, 3) for img in results['imgs']))

        # test Rand Augment
        rand_augment = PytorchVideoTrans(type='RandAugment')
        results = rand_augment(results)
        assert assert_dict_has_keys(results, target_keys)
        assert (all(img.shape == (32, 32, 3) for img in results['imgs']))
