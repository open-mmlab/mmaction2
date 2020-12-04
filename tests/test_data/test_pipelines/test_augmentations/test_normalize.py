import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import Normalize


class TestNormalize(TestAugumentations):

    @staticmethod
    def check_normalize(origin_imgs, result_imgs, norm_cfg):
        """Check if the origin_imgs are normalized correctly into result_imgs
        in a given norm_cfg."""
        target_imgs = result_imgs.copy()
        target_imgs *= norm_cfg['std']
        target_imgs += norm_cfg['mean']
        if norm_cfg['to_bgr']:
            target_imgs = target_imgs[..., ::-1].copy()
        assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)

    def test_normalize(self):
        with pytest.raises(TypeError):
            # mean must be list, tuple or np.ndarray
            Normalize(
                dict(mean=[123.675, 116.28, 103.53]), [58.395, 57.12, 57.375])

        with pytest.raises(TypeError):
            # std must be list, tuple or np.ndarray
            Normalize([123.675, 116.28, 103.53],
                      dict(std=[58.395, 57.12, 57.375]))

        target_keys = ['imgs', 'img_norm_cfg', 'modality']

        # normalize imgs in RGB format
        imgs = list(np.random.rand(2, 240, 320, 3).astype(np.float32))
        results = dict(imgs=imgs, modality='RGB')
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        # normalize flow imgs
        imgs = list(np.random.rand(4, 240, 320).astype(np.float32))
        results = dict(imgs=imgs, modality='Flow')
        config = dict(mean=[128, 128], std=[128, 128])
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        assert normalize_results['imgs'].shape == (2, 240, 320, 2)
        x_components = np.array(imgs[0::2])
        y_components = np.array(imgs[1::2])
        x_components = (x_components - config['mean'][0]) / config['std'][0]
        y_components = (y_components - config['mean'][1]) / config['std'][1]
        result_imgs = np.stack([x_components, y_components], axis=-1)
        assert np.all(np.isclose(result_imgs, normalize_results['imgs']))

        # normalize imgs in BGR format
        imgs = list(np.random.rand(2, 240, 320, 3).astype(np.float32))
        results = dict(imgs=imgs, modality='RGB')
        config = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=True)
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert self.check_keys_contain(normalize_results.keys(), target_keys)
        self.check_normalize(imgs, normalize_results['imgs'],
                             normalize_results['img_norm_cfg'])

        assert normalize.__repr__() == (
            normalize.__class__.__name__ +
            f'(mean={np.array([123.675, 116.28, 103.53])}, ' +
            f'std={np.array([58.395, 57.12, 57.375])}, to_bgr={True}, '
            f'adjust_magnitude={False})')
