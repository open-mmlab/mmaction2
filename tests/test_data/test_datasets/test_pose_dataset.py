import numpy as np
import pytest

from mmaction.datasets import PoseDataset
from .base import BaseTestDataset


class TestPoseDataset(BaseTestDataset):

    def test_pose_dataset(self):
        ann_file = self.pose_ann_file
        data_prefix = 'root'
        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            box_thr='0.5',
            data_prefix=data_prefix)
        assert len(dataset) == 100
        item = dataset[0]
        assert item['filename'].startswith(data_prefix)

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            valid_ratio=0.2,
            box_thr='0.9',
            data_prefix=data_prefix)
        assert len(dataset) == 84
        for item in dataset:
            assert item['filename'].startswith(data_prefix)
            assert np.all(item['box_score'][item['anno_inds']] >= 0.9)
            assert item['valid@0.9'] / item['total_frames'] >= 0.2

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            valid_ratio=0.3,
            box_thr='0.7',
            data_prefix=data_prefix)
        assert len(dataset) == 87
        for item in dataset:
            assert item['filename'].startswith(data_prefix)
            assert np.all(item['box_score'][item['anno_inds']] >= 0.7)
            assert item['valid@0.7'] / item['total_frames'] >= 0.3

        class_prob = {i: 1 for i in range(400)}
        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
            valid_ratio=0.3,
            box_thr='0.7',
            data_prefix=data_prefix,
            class_prob=class_prob)

        with pytest.raises(AssertionError):
            dataset = PoseDataset(
                ann_file=ann_file,
                pipeline=[],
                valid_ratio=0.2,
                box_thr='0.55',
                data_prefix=data_prefix)
