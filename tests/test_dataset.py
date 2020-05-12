import os.path as osp

import numpy as np
import pytest

from mmaction.datasets import RawframeDataset, RepeatDataset, VideoDataset


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

    def test_dataset_realpath(self):
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  '.')
        assert dataset.data_prefix == osp.realpath('.')
        dataset = RawframeDataset(self.frame_ann_file, self.frame_pipeline,
                                  's3://good')
        assert dataset.data_prefix == 's3://good'

    def test_video_dataset(self):
        video_dataset = VideoDataset(self.video_ann_file, self.video_pipeline,
                                     self.data_prefix)
        video_infos = video_dataset.video_infos
        video_filename = osp.join(self.data_prefix, 'test.mp4')
        assert video_infos == [dict(filename=video_filename, label=0)] * 2

    def test_rawframe_pipeline(self):
        target_keys = ['frame_dir', 'total_frames', 'label', 'filename_tmpl']

        # RawframeDataset not in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset in multi-class tasks
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=False)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # RawframeDataset in multi-class tasks in test mode
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=400,
            test_mode=True)
        result = rawframe_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

    def test_video_pipeline(self):
        target_keys = ['filename', 'label']

        # VideoDataset not in test mode
        video_dataset = VideoDataset(
            self.video_ann_file,
            self.video_pipeline,
            self.data_prefix,
            test_mode=False)
        result = video_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # VideoDataset in test mode
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
            # results must be a list
            rawframe_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            rawframe_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            rawframe_dataset.evaluate([0] * len(rawframe_dataset), topk=1.0)

        with pytest.raises(KeyError):
            # unsupported metric
            rawframe_dataset.evaluate(
                [0] * len(rawframe_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
        results = [np.array([0.1, 0.5, 0.4])] * 2
        eval_result = rawframe_dataset.evaluate(
            results, metrics=['top_k_accuracy', 'mean_class_accuracy'])
        assert set(eval_result.keys()) == set(
            ['top1_acc', 'top5_acc', 'mean_class_accuracy'])

    def test_video_evaluate(self):
        video_dataset = VideoDataset(self.video_ann_file, self.video_pipeline,
                                     self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            video_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            video_dataset.evaluate([0] * 5)

        with pytest.raises(TypeError):
            # topk must be int or tuple of int
            video_dataset.evaluate([0] * len(video_dataset), topk=1.0)

        with pytest.raises(KeyError):
            # unsupported metric
            video_dataset.evaluate([0] * len(video_dataset), metrics='iou')

        # evaluate top_k_accuracy and mean_class_accuracy metric
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

    def test_repeat_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        repeat_dataset = RepeatDataset(rawframe_dataset, 5)
        assert len(repeat_dataset) == 10
        result_a = repeat_dataset[0]
        result_b = repeat_dataset[2]
        assert set(result_a.keys()) == set(result_b.keys())
        for key in result_a:
            if isinstance(result_a[key], np.ndarray):
                assert np.equal(result_a[key], result_b[key]).all()
            elif isinstance(result_a[key], list):
                assert all(
                    np.array_equal(a, b)
                    for (a, b) in zip(result_a[key], result_b[key]))
            else:
                assert result_a[key] == result_b[key]
