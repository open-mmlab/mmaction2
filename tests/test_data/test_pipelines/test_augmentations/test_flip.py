import copy

import mmcv
import numpy as np
import pytest
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import Flip, Fuse


class TestFlip(TestAugumentations):

    @staticmethod
    def check_flip(origin_imgs, result_imgs, flip_type):
        """Check if the origin_imgs are flipped correctly into result_imgs in
        different flip_types."""
        n = len(origin_imgs)
        h, w, c = origin_imgs[0].shape
        if flip_type == 'horizontal':
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for channel in range(c):
                            if result_imgs[i][j, k, channel] != origin_imgs[i][j, w - 1 - k, channel]:  # noqa:E501
                                return False
            # yapf: enable
        else:
            # yapf: disable
            for i in range(n):
                for j in range(h):
                    for k in range(w):
                        for channel in range(c):
                            if result_imgs[i][j, k, channel] != origin_imgs[i][h - 1 - j, k, channel]:  # noqa:E501
                                return False
            # yapf: enable
        return True

    def test_flip(self):
        with pytest.raises(ValueError):
            # direction must be in ['horizontal', 'vertical']
            Flip(direction='vertically')

        target_keys = ['imgs', 'flip_direction', 'modality']

        # do not flip imgs.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=0, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        assert np.array_equal(imgs, results['imgs'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs horizontally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        if flip_results['flip'] is True:
            assert self.check_flip(imgs, flip_results['imgs'],
                                   flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # flip flow images horizontally
        imgs = [
            np.arange(16).reshape(4, 4).astype(np.float32),
            np.arange(16, 32).reshape(4, 4).astype(np.float32)
        ]
        results = dict(imgs=copy.deepcopy(imgs), modality='Flow')
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        imgs = [x.reshape(4, 4, 1) for x in imgs]
        flip_results['imgs'] = [
            x.reshape(4, 4, 1) for x in flip_results['imgs']
        ]
        if flip_results['flip'] is True:
            assert self.check_flip([imgs[0]],
                                   [mmcv.iminvert(flip_results['imgs'][0])],
                                   flip_results['flip_direction'])
            assert self.check_flip([imgs[1]], [flip_results['imgs'][1]],
                                   flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=1, direction='vertical')
        flip_results = flip(results)
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        if flip_results['flip'] is True:
            assert self.check_flip(imgs, flip_results['imgs'],
                                   flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        assert repr(flip) == (f'{flip.__class__.__name__}'
                              f'(flip_ratio={1}, direction=vertical, '
                              f'lazy={False})')

    def test_flip_lazy(self):
        with pytest.raises(ValueError):
            Flip(direction='vertically', lazy=True)

        target_keys = ['imgs', 'flip_direction', 'modality']

        # do not flip imgs.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=0, direction='horizontal', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert np.equal(imgs, results['imgs']).all()
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        # always flip imgs horizontally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=1, direction='horizontal', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        imgs_tmp = imgs.copy()
        results = dict(imgs=imgs_tmp, modality='RGB')
        flip = Flip(flip_ratio=1, direction='vertical', lazy=True)
        flip_results = flip(results)
        assert id(imgs_tmp) == id(flip_results['imgs'])
        assert self.check_keys_contain(flip_results.keys(), target_keys)
        flip_results_fuse = Fuse()(flip_results)
        assert self.check_flip(imgs, flip_results['imgs'],
                               flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert flip_results_fuse['imgs'][0].shape == (64, 64, 3)

        assert repr(flip) == (f'{flip.__class__.__name__}'
                              f'(flip_ratio={1}, direction=vertical, '
                              f'lazy={True})')
