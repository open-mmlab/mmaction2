import os.path as osp
import pickle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.core import (area2d, iou2d, iou3d, overlap2d,
                           spatio_temporal_iou3d, spatio_temporal_nms3d)


class TestBboxOverlaps:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), 'data', 'test_tubes')
        cls.tube_dir = osp.join(cls.data_prefix, 'tubes')
        cls.tube_file = osp.join(cls.data_prefix, 'tubes.npy')
        cls.det_file = osp.join(cls.data_prefix, 'all_dets.npy')
        cls.gt_tubes_file = osp.join(cls.data_prefix, 'gt_tubes.pkl')
        cls.tube = list(np.load(cls.tube_file, allow_pickle=True))
        cls.gt_tubes = pickle.load(open(cls.gt_tubes_file, 'rb'))
        cls.videos = ('Basketball/v_Basketball_g01_c01',
                      'BasketballDunk/v_BasketballDunk_g01_c01')
        cls.det_results = np.load(cls.det_file)

    @staticmethod
    def test_overlap2d():
        origin_box = np.array([[2, 2, 4, 4], [2, 2, 4, 4], [2, 2, 4, 4]])
        target_box = np.array([[3, 3, 5, 5], [2, 2, 5, 5], [4, 4, 6, 6]])

        overlap = overlap2d(origin_box, target_box)
        assert_array_equal(overlap, np.array([1, 4, 0]))

    @staticmethod
    def test_area2d():
        box = np.array([[2, 2, 4, 4], [2, 2, 2, 2], [4, 4, 2, 2]])
        area = area2d(box)
        assert_array_equal(area, np.array([4, 0, 4]))

    @staticmethod
    def test_iou2d():
        origin_boxes = np.array([[2, 2, 4, 4], [2, 2, 4, 4], [2, 2, 4, 4]])
        target_boxes = np.array([[3, 3, 5, 5], [2, 2, 5, 5], [4, 4, 6, 6]])
        gts = [1 / 7, 4 / 9, 0.]

        with pytest.raises(AssertionError):
            # box should be on 1 dimension
            iou2d(origin_boxes, target_boxes)

        for gt, origin_box, target_box in zip(gts, origin_boxes, target_boxes):
            iou = iou2d(origin_box, target_box)
            assert gt == iou

    @pytest.mark.parametrize('overlap, gt', [(0, (0, 1)), (0.25, (0, 1)),
                                             (0.5, (0, 1)), (1, (0, 1)),
                                             (-1, (0, ))])
    def test_spatio_temporal_nms3d(self, overlap, gt):
        index = spatio_temporal_nms3d(self.tube, overlap)
        assert (index == gt).all()

    @pytest.mark.parametrize('spatial_only, gt', [(True, 0.7599830031394958),
                                                  (False, 0.2371578165825377)])
    def test_spatio_temporal_iou3d(self, spatial_only, gt):
        video = self.videos[0]
        tube_path = osp.join(self.tube_dir, video + '_tubes.pkl')
        tube = pickle.load(open(tube_path, 'rb'))[0][0][:-1][0]
        gt_tube = self.gt_tubes[video][0][0]
        result = spatio_temporal_iou3d(gt_tube, tube, spatial_only)
        assert_array_almost_equal(result, gt)

    def test_iou3d(self):
        with pytest.raises(AssertionError):
            box1 = np.array([[0, 1, 2, 3, 4]])
            box2 = np.array([[1, 1, 2, 3, 4]])
            iou3d(box1, box2)

        with pytest.raises(AssertionError):
            box1 = np.array([[0, 1, 2, 3, 4]])
            box2 = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
            iou3d(box1, box2)

        box1 = np.array([[9, 188, 101, 235, 179], [10, 190, 103, 237, 181],
                         [11, 192, 105, 239, 183]])
        box2 = np.array([[9, 178, 91, 225, 169], [10, 180, 93, 227, 171],
                         [11, 182, 95, 229, 173]])
        assert_array_almost_equal(iou3d(box1, box2), 0.5224252491694352)

        box2 = np.array([[9, 188, 101, 235, 179], [10, 190, 103, 237, 181],
                         [11, 192, 105, 239, 183]])
        assert_array_almost_equal(iou3d(box1, box2), 1.0)

        box2 = np.array([[9, 1, 2, 3, 4], [10, 2, 3, 4, 5], [11, 6, 7, 8, 9]])
        assert_array_almost_equal(iou3d(box1, box2), 0.0)
