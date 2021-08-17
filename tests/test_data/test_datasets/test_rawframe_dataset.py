# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets import RawframeDataset
from .base import BaseTestDataset


class TestRawframDataset(BaseTestDataset):

    def test_rawframe_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, total_frames=5, label=127)
        ] * 2
        assert rawframe_dataset.start_index == 1

    def test_rawframe_dataset_with_offset(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, offset=2, total_frames=5, label=127)
        ] * 2
        assert rawframe_dataset.start_index == 1

    def test_rawframe_dataset_multi_label(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_multi_label,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=100)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'imgs')
        label0 = [1]
        label1 = [3, 5]
        labels = [label0, label1]
        for info, label in zip(rawframe_infos, labels):
            assert info['frame_dir'] == frame_dir
            assert info['total_frames'] == 5
            assert set(info['label']) == set(label)
        assert rawframe_dataset.start_index == 1

    def test_dataset_realpath(self):
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  '.')
        assert dataset.data_prefix == osp.realpath('.')
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  's3://good')
        assert dataset.data_prefix == 's3://good'

        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline)
        assert dataset.data_prefix is None
        assert dataset.video_infos[0]['frame_dir'] == 'imgs'

    def test_rawframe_pipeline(self):
        target_keys = [
            'frame_dir', 'total_frames', 'label', 'filename_tmpl',
            'start_index', 'modality'
        ]

        # RawframeDataset not in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset in multi-class tasks
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys + ['offset'])

        # RawframeDataset in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset in multi-class tasks in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # RawframeDataset with offset
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert assert_dict_has_keys(result, target_keys + ['offset'])

    def test_rawframe_evaluate(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            rawframe_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            rawframe_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = rawframe_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])
