import copy

import mmcv
import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets.pipelines import Flip
from .base import check_flip


class TestFlip:

    def test_flip(self):
        with pytest.raises(ValueError):
            # direction must be in ['horizontal', 'vertical']
            Flip(direction='vertically')

        target_keys = ['imgs', 'flip_direction', 'modality']

        # do not flip imgs.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        results['gt_bboxes'] = np.array([[0, 0, 60, 60]])
        results['proposals'] = None
        flip = Flip(flip_ratio=0, direction='horizontal')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        assert np.array_equal(imgs, results['imgs'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs horizontally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=1, direction='horizontal')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        if flip_results['flip'] is True:
            assert check_flip(imgs, flip_results['imgs'],
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
        assert assert_dict_has_keys(flip_results, target_keys)
        imgs = [x.reshape(4, 4, 1) for x in imgs]
        flip_results['imgs'] = [
            x.reshape(4, 4, 1) for x in flip_results['imgs']
        ]
        if flip_results['flip'] is True:
            assert check_flip([imgs[0]],
                              [mmcv.iminvert(flip_results['imgs'][0])],
                              flip_results['flip_direction'])
            assert check_flip([imgs[1]], [flip_results['imgs'][1]],
                              flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        # always flip imgs vertivally.
        imgs = list(np.random.rand(2, 64, 64, 3))
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB')
        flip = Flip(flip_ratio=1, direction='vertical')
        flip_results = flip(results)
        assert assert_dict_has_keys(flip_results, target_keys)
        if flip_results['flip'] is True:
            assert check_flip(imgs, flip_results['imgs'],
                              flip_results['flip_direction'])
        assert id(flip_results['imgs']) == id(results['imgs'])
        assert np.shape(flip_results['imgs']) == np.shape(imgs)

        assert repr(flip) == (f'{flip.__class__.__name__}'
                              f'(flip_ratio={1}, direction=vertical, '
                              f'flip_label_map={None}, lazy={False})')

        # transform label for the flipped image with the specific label.
        _flip_label_map = {4: 6}
        imgs = list(np.random.rand(2, 64, 64, 3))

        # the label should be mapped.
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', label=4)
        flip = Flip(
            flip_ratio=1,
            direction='horizontal',
            flip_label_map=_flip_label_map)
        flip_results = flip(results)
        assert results['label'] == 6

        # the label should not be mapped.
        results = dict(imgs=copy.deepcopy(imgs), modality='RGB', label=3)
        flip = Flip(
            flip_ratio=1,
            direction='horizontal',
            flip_label_map=_flip_label_map)
        flip_results = flip(results)
        assert results['label'] == 3
