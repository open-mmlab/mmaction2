import os.path as osp

import numpy as np
import pytest

from mmaction.datasets import (AudioDataset, AudioFeatureDataset,
                               AudioVisualDataset)
from .test_base_dataset import TestBaseDataset


class TestAudioDataset(TestBaseDataset):

    def test_audio_dataset(self):
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix)
        audio_infos = audio_dataset.video_infos
        wav_path = osp.join(self.data_prefix, 'test.wav')
        assert audio_infos == [
            dict(audio_path=wav_path, total_frames=100, label=127)
        ] * 2

    def test_audio_pipeline(self):
        target_keys = [
            'audio_path', 'label', 'start_index', 'modality', 'audios_shape',
            'length', 'sample_rate', 'total_frames'
        ]

        # Audio dataset not in test mode
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix,
            test_mode=False)
        result = audio_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # Audio dataset in test mode
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = audio_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_audio_evaluate(self):
        audio_dataset = AudioDataset(
            self.audio_ann_file,
            self.audio_pipeline,
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
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])


class TestAudioFeatureDataset(TestBaseDataset):

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
        assert self.check_keys_contain(result.keys(), target_keys)

        # Audio dataset in test mode
        audio_feature_dataset = AudioFeatureDataset(
            self.audio_feature_ann_file,
            self.audio_feature_pipeline,
            data_prefix=self.data_prefix,
            test_mode=True)
        result = audio_feature_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

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


class TestAudioVisualDataset(TestBaseDataset):

    def test_audio_visual_dataset(self):
        test_dataset = AudioVisualDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            video_prefix=self.data_prefix,
            data_prefix=self.data_prefix)
        video_infos = test_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        audio_path = osp.join(self.data_prefix, 'test_imgs.npy')
        filename = osp.join(self.data_prefix, 'test_imgs.mp4')
        assert video_infos == [
            dict(
                frame_dir=frame_dir,
                audio_path=audio_path,
                filename=filename,
                total_frames=5,
                label=127)
        ] * 2
        assert test_dataset.start_index == 1
