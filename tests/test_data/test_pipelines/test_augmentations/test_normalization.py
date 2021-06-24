import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import Normalize
from .base import check_normalize


class TestNormalization:

    @staticmethod
    def test_normalize():
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
        assert assert_dict_has_keys(normalize_results, target_keys)
        check_normalize(imgs, normalize_results['imgs'],
                        normalize_results['img_norm_cfg'])

        # normalize flow imgs
        imgs = list(np.random.rand(4, 240, 320).astype(np.float32))
        results = dict(imgs=imgs, modality='Flow')
        config = dict(mean=[128, 128], std=[128, 128])
        normalize = Normalize(**config)
        normalize_results = normalize(results)
        assert assert_dict_has_keys(normalize_results, target_keys)
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
        assert assert_dict_has_keys(normalize_results, target_keys)
        check_normalize(imgs, normalize_results['imgs'],
                        normalize_results['img_norm_cfg'])

        assert normalize.__repr__() == (
            normalize.__class__.__name__ +
            f'(mean={np.array([123.675, 116.28, 103.53])}, ' +
            f'std={np.array([58.395, 57.12, 57.375])}, to_bgr={True}, '
            f'adjust_magnitude={False})')
