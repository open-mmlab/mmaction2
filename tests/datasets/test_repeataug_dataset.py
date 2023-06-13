# Copyright (c) OpenMMLab. All rights reserved.
import pytest
from mmengine.testing import assert_dict_has_keys

from mmaction.datasets import RepeatAugDataset
from mmaction.utils import register_all_modules
from .base import BaseTestDataset


class TestVideoDataset(BaseTestDataset):
    register_all_modules()

    def test_video_dataset(self):
        with pytest.raises(AssertionError):
            # Currently only support decord backend
            video_dataset = RepeatAugDataset(
                self.video_ann_file,
                self.video_pipeline,
                data_prefix={'video': self.data_prefix},
                start_index=3)

        video_pipeline = [
            dict(type='DecordInit'),
            dict(
                type='SampleFrames', clip_len=4, frame_interval=2,
                num_clips=1),
            dict(type='DecordDecode')
        ]

        video_dataset = RepeatAugDataset(
            self.video_ann_file,
            video_pipeline,
            data_prefix={'video': self.data_prefix},
            start_index=3)
        assert len(video_dataset) == 2
        assert video_dataset.start_index == 3

        video_dataset = RepeatAugDataset(
            self.video_ann_file,
            video_pipeline,
            data_prefix={'video': self.data_prefix})
        assert video_dataset.start_index == 0

    def test_video_dataset_multi_label(self):
        video_pipeline = [
            dict(type='DecordInit'),
            dict(
                type='SampleFrames', clip_len=4, frame_interval=2,
                num_clips=1),
            dict(type='DecordDecode')
        ]
        video_dataset = RepeatAugDataset(
            self.video_ann_file_multi_label,
            video_pipeline,
            data_prefix={'video': self.data_prefix},
            multi_class=True,
            num_classes=100)
        assert video_dataset.start_index == 0

    def test_video_pipeline(self):
        video_pipeline = [
            dict(type='DecordInit'),
            dict(
                type='SampleFrames', clip_len=4, frame_interval=2,
                num_clips=1),
            dict(type='DecordDecode')
        ]
        target_keys = ['filename', 'label', 'start_index', 'modality']

        # RepeatAugDataset not in test mode
        video_dataset = RepeatAugDataset(
            self.video_ann_file,
            video_pipeline,
            data_prefix={'video': self.data_prefix})
        result = video_dataset[0]
        assert isinstance(result, (list, tuple))
        assert assert_dict_has_keys(result[0], target_keys)
