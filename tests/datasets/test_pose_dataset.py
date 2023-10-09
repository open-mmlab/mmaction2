# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmaction.datasets import PoseDataset
from .base import BaseTestDataset


class TestPoseDataset(BaseTestDataset):

    def test_pose_dataset(self):
        ann_file = self.pose_ann_file
        data_prefix = dict(video='root')
        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            split='train',
            box_thr=0.5,
            data_prefix=data_prefix)
        assert len(dataset) == 100
        item = dataset[0]
        assert item['frame_dir'].startswith(data_prefix['video'])

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            split='train',
            valid_ratio=0.2,
            box_thr=0.9)
        assert len(dataset) == 84
        for item in dataset:
            assert np.all(item['box_score'][item['anno_inds']] >= 0.9)
            assert item['valid'][0.9] / item['total_frames'] >= 0.2

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            split='train',
            valid_ratio=0.3,
            box_thr=0.7)
        assert len(dataset) == 87
        for item in dataset:
            assert np.all(item['box_score'][item['anno_inds']] >= 0.7)
            assert item['valid'][0.7] / item['total_frames'] >= 0.3

        with pytest.raises(AssertionError):
            dataset = PoseDataset(
                ann_file=ann_file, pipeline=[], valid_ratio=0.2, box_thr=0.55)
