# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.testing import assert_dict_has_keys

from mmaction.datasets import VideoTextDataset
from mmaction.utils import register_all_modules
from .base import BaseTestDataset


class TestVideoTextDataset(BaseTestDataset):
    register_all_modules()

    def test_video_dataset(self):
        video_dataset = VideoTextDataset(
            self.video_text_ann_file,
            self.video_text_pipeline,
            data_prefix={'video': self.data_prefix},
            start_index=3)
        assert len(video_dataset) == 2
        assert video_dataset.start_index == 3

        video_dataset = VideoTextDataset(
            self.video_text_ann_file,
            self.video_text_pipeline,
            data_prefix={'video': self.data_prefix})
        assert video_dataset.start_index == 0

    def test_video_pipeline(self):
        target_keys = ['filename', 'text', 'start_index', 'modality', 'imgs']

        # VideoTextDataset not in test mode
        video_dataset = VideoTextDataset(
            self.video_text_ann_file,
            self.video_text_pipeline,
            data_prefix={'video': self.data_prefix},
            test_mode=False)
        result = video_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # VideoTextDataset in test mode
        video_dataset = VideoTextDataset(
            self.video_text_ann_file,
            self.video_text_pipeline,
            data_prefix={'video': self.data_prefix},
            test_mode=True)
        result = video_dataset[0]
        assert assert_dict_has_keys(result, target_keys)
