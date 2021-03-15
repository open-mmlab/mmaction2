import os.path as osp

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets import VideoDataset
from .base import BaseTestDataset


class TestVideoDataset(BaseTestDataset):

    def test_video_dataset(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            start_index=3)
        assert len(video_dataset) == 2
        assert video_dataset.start_index == 3

        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        assert video_infos == [dict(filename=video_filename, label=0)] * 2
        assert video_dataset.start_index == 0

    def test_video_dataset_multi_label(self):
        video_dataset = VideoDataset(
            self.video_ann_file_multi_label,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            multi_class=True,
            num_classes=100)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        label0 = [0, 3]
        label1 = [0, 2, 4]
        labels = [label0, label1]
        for info, label in zip(video_infos, labels):
            print(info, video_filename)
            assert info['filename'] == video_filename
            assert set(info['label']) == set(label)
        assert video_dataset.start_index == 0

    def test_video_pipeline(self):
        target_keys = ['filename', 'label', 'start_index', 'modality']

        # VideoDataset not in test mode
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = video_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # VideoDataset in test mode
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = video_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

    def test_video_evaluate(self):
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            video_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            video_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            video_dataset.evaluate(
                [0] * len(video_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            video_dataset.evaluate([0] * len(video_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = video_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])
