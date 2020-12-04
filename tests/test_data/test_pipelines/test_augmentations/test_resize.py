import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import Fuse, Resize


class TestResize(TestAugumentations):

    def test_resize(self):
        with pytest.raises(ValueError):
            # scale must be positive
            Resize(-0.5)

        with pytest.raises(TypeError):
            # scale must be tuple of int
            Resize('224')

        target_keys = [
            'imgs', 'img_shape', 'keep_ratio', 'scale_factor', 'modality'
        ]

        # test resize for flow images
        imgs = list(np.random.rand(2, 240, 320))
        results = dict(imgs=imgs, modality='Flow')
        resize = Resize(scale=(160, 80), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [.5, 1. / 3.], dtype=np.float32))
        assert resize_results['img_shape'] == (80, 160)

        # scale with -1 to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(-1, 256), keep_ratio=True)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(320, 320), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [1, 320 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(341, 256), keep_ratio=False)
        resize_results = resize(results)
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        assert np.all(resize_results['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results['img_shape'] == (256, 341)

        assert repr(resize) == (
            resize.__class__.__name__ +
            f'(scale={(341, 256)}, keep_ratio={False}, ' +
            f'interpolation=bilinear, lazy={False})')

    def test_resize_lazy(self):
        with pytest.raises(ValueError):
            # scale must be positive
            Resize(-0.5, lazy=True)

        with pytest.raises(TypeError):
            # scale must be tuple of int
            Resize('224', lazy=True)

        target_keys = [
            'imgs', 'img_shape', 'keep_ratio', 'scale_factor', 'modality'
        ]

        # scale with -1 to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(-1, 256), keep_ratio=True, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (256, 341)

        # scale with a normal tuple (320, 320) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(320, 320), keep_ratio=False, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [1, 320 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (320, 320)

        # scale with a normal tuple (341, 256) to indicate np.inf
        imgs = list(np.random.rand(2, 240, 320, 3))
        results = dict(imgs=imgs, modality='RGB')
        resize = Resize(scale=(341, 256), keep_ratio=False, lazy=True)
        resize_results = resize(results)
        assert id(imgs) == id(resize_results['imgs'])
        assert self.check_keys_contain(resize_results.keys(), target_keys)
        resize_results_fuse = Fuse()(resize_results)
        assert np.all(resize_results_fuse['scale_factor'] == np.array(
            [341 / 320, 256 / 240], dtype=np.float32))
        assert resize_results_fuse['img_shape'] == (256, 341)

        assert repr(resize) == (f'{resize.__class__.__name__ }'
                                f'(scale={(341, 256)}, keep_ratio={False}, ' +
                                f'interpolation=bilinear, lazy={True})')
