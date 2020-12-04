import copy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tests.test_data.test_pipelines.test_augmentations.test_base_aug import \
    TestAugumentations

from mmaction.datasets.pipelines import (EntityBoxClip, EntityBoxCrop,
                                         EntityBoxFlip, EntityBoxPad,
                                         EntityBoxRescale)


class TestBoxes(TestAugumentations):

    def test_box_rescale(self):
        target_keys = ['img_shape', 'scale_factor', 'ann', 'proposals']
        results = dict(
            img_shape=(520, 480),
            scale_factor=(0.7, 0.8),
            proposals=np.array([[0.011, 0.157, 0.655, 0.983, 0.998163]]),
            ann=dict(entity_boxes=np.array([[0.031, 0.162, 0.67, 0.995]])))

        with pytest.raises(AssertionError):
            box_scale = EntityBoxRescale()
            results_ = copy.deepcopy(results)
            results_['proposals'] = np.array([[0.011, 0.157, 0.655]])
            box_scale(results_)

        box_scale = EntityBoxRescale()
        results_ = copy.deepcopy(results)
        results_ = box_scale(results_)
        self.check_keys_contain(results_.keys(), target_keys + ['scores'])
        assert_array_almost_equal(
            results_['proposals'],
            np.array([[3.696000, 65.311999, 220.079995, 408.928002]]))
        assert_array_almost_equal(
            results_['ann']['entity_boxes'],
            np.array([[10.416000, 67.391998, 225.120004, 413.920019]]))
        assert results_['scores'] == np.array([0.998163], dtype=np.float32)

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_scale(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert results_['proposals'] is None

    def test_box_crop(self):
        target_keys = ['ann', 'proposals', 'crop_bbox']
        results = dict(
            proposals=np.array([[3.696000, 65.311999, 220.079995,
                                 408.928002]]),
            crop_bbox=[13, 75, 200, 380],
            ann=dict(
                entity_boxes=np.array(
                    [[10.416000, 67.391998, 225.120004, 413.920019]])))

        box_crop = EntityBoxCrop()
        results_ = copy.deepcopy(results)
        results_ = box_crop(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_almost_equal(
            results_['ann']['entity_boxes'],
            np.array([[-2.584, -7.608002, 212.120004, 338.920019]]))
        assert_array_almost_equal(
            results_['proposals'],
            np.array([[-9.304, -9.688001, 207.079995, 333.928002]]))

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_crop(results_)
        assert results_['proposals'] is None

    def test_box_flip(self):
        target_keys = ['ann', 'proposals', 'img_shape']
        results = dict(
            proposals=np.array([[-9.304, -9.688001, 207.079995, 333.928002]]),
            img_shape=(520, 480),
            ann=dict(
                entity_boxes=np.array(
                    [[-2.584, -7.608002, 212.120004, 338.920019]])))

        with pytest.raises(ValueError):
            EntityBoxFlip(0, 'unsupport')

        box_flip = EntityBoxFlip(flip_ratio=1)
        results_ = copy.deepcopy(results)
        results_ = box_flip(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_almost_equal(
            results_['ann']['entity_boxes'],
            np.array([[266.879996, -7.608002, 481.584, 338.920019]]))
        assert_array_almost_equal(
            results_['proposals'],
            np.array([[271.920005, -9.688001, 488.304, 333.928002]]))

        box_flip = EntityBoxFlip(flip_ratio=1, direction='vertical')
        results_ = copy.deepcopy(results)
        results_ = box_flip(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_almost_equal(
            results_['ann']['entity_boxes'],
            np.array([[-2.584, 180.079981, 212.120004, 526.608002]]))
        assert_array_almost_equal(
            results_['proposals'],
            np.array([[-9.304, 185.071998, 207.079995, 528.688001]]))

        box_flip = EntityBoxFlip()
        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_flip(results_)
        assert results_['proposals'] is None

        assert repr(box_flip) == (f'{box_flip.__class__.__name__}'
                                  '(flip_ratio=0.5, direction=horizontal)')

    def test_box_clip(self):
        target_keys = ['ann', 'proposals', 'img_shape']
        results = dict(
            proposals=np.array([[-9.304, -9.688001, 207.079995, 333.928002]]),
            img_shape=(335, 210),
            ann=dict(
                entity_boxes=np.array(
                    [[-2.584, -7.608002, 212.120004, 338.920019]])))

        box_clip = EntityBoxClip()
        results_ = copy.deepcopy(results)
        results_ = box_clip(results_)

        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_equal(results_['ann']['entity_boxes'],
                           np.array([[0., 0., 209., 334.]]))
        assert_array_equal(results_['proposals'],
                           np.array([[0., 0., 207.079995, 333.928002]]))

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_clip(results_)
        assert results_['proposals'] is None

    def test_box_pad(self):
        target_keys = ['ann', 'proposals', 'img_shape']
        results = dict(
            proposals=np.array([[-9.304, -9.688001, 207.079995, 333.928002],
                                [-2.584, -7.608002, 212.120004, 338.920019]]),
            img_shape=(335, 210),
            ann=dict(
                entity_boxes=np.array([[
                    -2.584, -7.608002, 212.120004, 338.920019
                ], [-9.304, -9.688001, 207.079995, 333.928002]])))

        box_pad_none = EntityBoxPad()
        results_ = copy.deepcopy(results)
        results_ = box_pad_none(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_equal(results_['proposals'], results['proposals'])
        assert_array_equal(results_['ann']['entity_boxes'],
                           results['ann']['entity_boxes'])

        box_pad = EntityBoxPad(3)
        results_ = copy.deepcopy(results)
        results_ = box_pad(results_)
        self.check_keys_contain(results_.keys(), target_keys)
        assert_array_equal(
            results_['proposals'],
            np.array([[-9.304, -9.688001, 207.079995, 333.928002],
                      [-2.584, -7.608002, 212.120004, 338.920019],
                      [0., 0., 0., 0.]],
                     dtype=np.float32))
        assert_array_equal(
            results_['ann']['entity_boxes'],
            np.array([[-2.584, -7.608002, 212.120004, 338.920019],
                      [-9.304, -9.688001, 207.079995, 333.928002],
                      [0., 0., 0., 0.]],
                     dtype=np.float32))

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_pad(results_)
        assert results_['proposals'] is None

        assert repr(box_pad) == (f'{box_pad.__class__.__name__}'
                                 '(max_num_gts=3)')
