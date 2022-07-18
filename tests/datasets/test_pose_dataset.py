# Copyright (c) OpenMMLab. All rights reserved.
from mmaction.datasets import PoseDataset
from .base import BaseTestDataset


class TestPoseDataset(BaseTestDataset):

    def test_pose_dataset(self):
        ann_file = self.pose_ann_file

        dataset = PoseDataset(
            ann_file=ann_file,
            pipeline=[],
        )
        assert len(dataset) == 100
