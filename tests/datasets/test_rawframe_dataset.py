# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.testing import assert_dict_has_keys

from mmaction.datasets import RawframeDataset
from mmaction.utils import register_all_modules
from .base import BaseTestDataset


class TestRawframDataset(BaseTestDataset):

    def test_rawframe_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           {'img': self.data_prefix})
        assert rawframe_dataset.start_index == 1

    def test_rawframe_dataset_with_offset(self):
        register_all_modules()
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline, {'img': self.data_prefix},
            with_offset=True)
        assert rawframe_dataset.start_index == 1

    def test_rawframe_dataset_multi_label(self):
        register_all_modules()
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_multi_label,
            self.frame_pipeline, {'img': self.data_prefix},
            multi_class=True,
            num_classes=100)
        assert rawframe_dataset.start_index == 1

    def test_dataset_realpath(self):
        register_all_modules()
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  {'img': '.'})
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  {'img': 's3://good'})
        assert dataset.data_prefix == {'img': 's3://good'}

        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline)

    def test_rawframe_pipeline(self):
        target_keys = [
            'frame_dir', 'total_frames', 'label', 'filename_tmpl',
            'start_index', 'modality'
        ]

        # RawframeDataset not in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline, {'img': self.data_prefix},
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset in multi-class tasks
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline, {'img': self.data_prefix},
            multi_class=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline, {'img': self.data_prefix},
            with_offset=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys + ['offset'])

        # RawframeDataset in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline, {'img': self.data_prefix},
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset in multi-class tasks in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline, {'img': self.data_prefix},
            multi_class=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline, {'img': self.data_prefix},
            with_offset=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys + ['offset'])
