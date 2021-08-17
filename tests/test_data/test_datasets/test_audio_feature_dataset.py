# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import numpy as np
import pytest
from mmcv.utils import assert_dict_has_keys

from mmaction.datasets import AudioFeatureDataset
from .base import BaseTestDataset


class TestAudioFeatureDataset(BaseTestDataset):

    def test_audio_feature_dataset(self):
        audio_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix)
        audio_infos = audio_dataset.video_infos
        feature_path = osp.join(self.data_prefix, 'test.npy')
        assert audio_infos == [
            dict(audio_path=feature_path, total_frames=100, label=127)
        ] * 2

    def test_audio_feature_pipeline(self):
        target_keys = [
            'audio_path', 'label', 'start_index', 'modality', 'audios',
            'total_frames'
        ]

        # Audio feature dataset not in test mode
        audio_feature_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = audio_feature_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        # Audio dataset in test mode
        audio_feature_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = audio_feature_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

    def test_audio_feature_evaluate(self):
        audio_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            audio_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            audio_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            audio_dataset.evaluate(
                [0] * len(audio_dataset),
                metric_options=dict(top_k_accuracy=dict(topk=1.)))

        with pytest.raises(KeyError):
            # unsupported metric
            audio_dataset.evaluate([0] * len(audio_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = audio_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])
