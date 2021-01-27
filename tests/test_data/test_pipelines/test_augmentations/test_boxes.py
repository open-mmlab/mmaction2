import copy

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal

from mmaction.datasets.pipelines import (EntityBoxCrop, EntityBoxFlip,
                                         EntityBoxRescale)


class TestBoxes:

    def test_box_crop(self):
        target_keys = ['proposals', 'crop_bbox', 'gt_bboxes']
        results = dict(
            proposals=np.array([[3.696, 65.312, 220.08, 408.928]]),
            crop_bbox=[13, 75, 200, 450],
            gt_bboxes=np.array([[10.416, 67.392, 225.12, 413.92]]))

        crop_bbox = results['crop_bbox']

        box_crop = EntityBoxCrop(crop_bbox)

        results_ = copy.deepcopy(results)
        results_ = box_crop(results_)
        assert_dict_has_keys(results_, target_keys)
        assert_array_almost_equal(results_['gt_bboxes'],
                                  np.array([[0, 0, 186, 338.92]]))
        assert_array_almost_equal(results_['proposals'],
                                  np.array([[0, 0, 186, 333.928]]))

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_crop(results_)
        assert results_['proposals'] is None
        assert repr(box_crop) == f'EntityBoxCrop(crop_bbox={crop_bbox})'

    def test_box_flip(self):
        target_keys = ['gt_bboxes', 'proposals', 'img_shape']
        results = dict(
            proposals=np.array([[0, 0, 186, 333.928]]),
            img_shape=(305, 200),
            gt_bboxes=np.array([[0, 0, 186, 338.92]]))

        img_shape = results['img_shape']

        box_flip = EntityBoxFlip(img_shape)
        results_ = copy.deepcopy(results)
        results_ = box_flip(results_)
        assert_dict_has_keys(results_, target_keys)
        assert_array_almost_equal(results_['gt_bboxes'],
                                  np.array([[13, 0, 199, 338.92]]))
        assert_array_almost_equal(results_['proposals'],
                                  np.array([[13, 0, 199, 333.928]]))

        box_flip = EntityBoxFlip(img_shape)
        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_flip(results_)
        assert results_['proposals'] is None
        assert repr(box_flip) == f'EntityBoxFlip(img_shape={img_shape})'

    def test_box_rescale(self):
        target_keys = ['img_shape', 'scale_factor', 'proposals', 'gt_bboxes']
        results = dict(
            img_shape=(520, 480),
            scale_factor=(0.7, 0.8),
            proposals=np.array([[5.28, 81.64, 314.4, 511.16]]),
            gt_bboxes=np.array([[14.88, 84.24, 321.6, 517.4]]))
        scale_factor = results['scale_factor']

        with pytest.raises(AssertionError):
            box_scale = EntityBoxRescale(scale_factor)
            results_ = copy.deepcopy(results)
            results_['proposals'] = np.array([[5.28, 81.64, 314.4]])
            box_scale(results_)

        box_scale = EntityBoxRescale(scale_factor)
        results_ = copy.deepcopy(results)
        results_ = box_scale(results_)
        assert_dict_has_keys(results_, target_keys)
        assert_array_almost_equal(results_['proposals'],
                                  np.array([[3.696, 65.312, 220.08, 408.928]]))
        assert_array_almost_equal(results_['gt_bboxes'],
                                  np.array([[10.416, 67.392, 225.12, 413.92]]))

        results_ = copy.deepcopy(results)
        results_['proposals'] = None
        results_ = box_scale(results_)
        assert_dict_has_keys(results_, target_keys)
        assert results_['proposals'] is None
        assert repr(box_scale) == ('EntityBoxRescale'
                                   f'(scale_factor={scale_factor})')
