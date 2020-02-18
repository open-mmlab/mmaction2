import os.path as osp

import numpy as np
import pytest

from mmaction.datasets import RawframeDataset, VideoDataset


class TestDataset(object):

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(__file__), 'data')
        cls.frame_ann_file = osp.join(cls.data_prefix, 'frame_test_list.txt')
        cls.video_ann_file = osp.join(cls.data_prefix, 'video_test_list.txt')

        cls.frame_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='FrameSelector', io_backend='disk')
        ]
        cls.video_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='OpenCVDecode')
        ]

    def test_rawframe_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, total_frames=5, label=127)
        ] * 2

    def test_video_dataset(self):
        video_dataset = VideoDataset(self.video_ann_file, self.video_pipeline,
                                     self.data_prefix)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        assert video_infos == [dict(filename=video_filename, label=0)] * 2

    def test_rawframe_pipeline(self):
        target_keys = ['frame_dir', 'total_frames', 'label', 'filename_tmpl']

        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_video_pipeline(self):
        target_keys = ['filename', 'label']

        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            self.data_prefix,
            test_mode=False)
        result = video_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            self.data_prefix,
            test_mode=True)
        result = video_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_rawframe_evaluate(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)

        with pytest.raises(TypeError):
            rawframe_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            rawframe_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            rawframe_dataset.evaluate([0] * len(rawframe_dataset), topk=1.0)

        with pytest.raises(KeyError):
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset), metrics='iou')

        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = rawframe_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

    def test_video_evaluate(self):
        video_dataset = VideoDataset(self.video_ann_file, self.video_pipeline,
                                     self.data_prefix)

        with pytest.raises(TypeError):
            video_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            video_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            video_dataset.evaluate([0] * len(video_dataset), topk=1.0)

        with pytest.raises(KeyError):
            video_dataset.evaluate([0] * len(video_dataset), metrics='iou')

        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = video_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

    def test_base_dataset(self):
        video_dataset = VideoDataset(self.video_ann_file, self.video_pipeline,
                                     self.data_prefix)
        assert len(video_dataset) == 2
        assert type(video_dataset[0]) == dict
