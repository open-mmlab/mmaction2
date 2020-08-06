import os
import os.path as osp
import tempfile

import mmcv
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_equal

from mmaction.datasets import (ActivityNetDataset, RawframeDataset,
                               RepeatDataset, VideoDataset)


class TestDataset(object):

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(osp.dirname(osp.dirname(__file__)), 'data')
        cls.frame_ann_file = osp.join(cls.data_prefix, 'frame_test_list.txt')
        cls.frame_ann_file_with_offset = osp.join(
            cls.data_prefix, 'frame_test_list_with_offset.txt')
        cls.frame_ann_file_multi_label = osp.join(
            cls.data_prefix, 'frame_test_list_multi_label.txt')
        cls.video_ann_file = osp.join(cls.data_prefix, 'video_test_list.txt')
        cls.action_ann_file = osp.join(cls.data_prefix,
                                       'action_test_anno.json')

        cls.frame_pipeline = [
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='FrameSelector', io_backend='disk')
        ]
        cls.video_pipeline = [
            dict(type='OpenCVInit'),
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1),
            dict(type='OpenCVDecode')
        ]
        cls.action_pipeline = []

    def test_rawframe_dataset(self):
        rawframe_dataset = RawframeDataset(self.frame_ann_file,
                                           self.frame_pipeline,
                                           self.data_prefix)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, total_frames=5, label=127)
        ] * 2

    def test_rawframe_dataset_with_offset(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_with_offset,
            self.frame_pipeline,
            self.data_prefix,
            with_offset=True)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        assert rawframe_infos == [
            dict(frame_dir=frame_dir, offset=2, total_frames=5, label=127)
        ] * 2

    def test_rawframe_dataset_multi_label(self):
        rawframe_dataset = RawframeDataset(
            self.frame_ann_file_multi_label,
            self.frame_pipeline,
            self.data_prefix,
            multi_class=True,
            num_classes=100)
        rawframe_infos = rawframe_dataset.video_infos
        frame_dir = osp.join(self.data_prefix, 'test_imgs')
        label0 = torch.zeros(100)
        label0[[1]] = 1.0
        label1 = torch.zeros(100)
        label1[[3, 5]] = 1.0
        labels = [label0, label1]
        for info, label in zip(rawframe_infos, labels):
            assert info['frame_dir'] == frame_dir
            assert info['total_frames'] == 5
            assert torch.all(info['label'] == label)

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

    def test_action_pipeline(self):
        target_keys = ['video_name', 'data_prefix']

        # ActivityNet Dataset not in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=False)
        result = action_dataset[0]
        assert self.check_keys_contain(result.keys(), target_keys)

        # ActivityNet Dataset in test mode
        action_dataset = ActivityNetDataset(
            self.action_ann_file,
            self.action_pipeline,
            self.data_prefix,
            test_mode=True)
        result = action_dataset[0]
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

    def test_activitynet_dataset(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        activitynet_infos = activitynet_dataset.video_infos
        assert activitynet_infos == [
            dict(
                video_name='v_test1',
                duration_second=1,
                duration_frame=30,
                annotations=[dict(segment=[0.3, 0.6], label='Rock climbing')],
                feature_frame=30,
                fps=30.0,
                rfps=30),
            dict(
                video_name='v_test2',
                duration_second=2,
                duration_frame=48,
                annotations=[dict(segment=[1.0, 2.0], label='Drinking beer')],
                feature_frame=48,
                fps=24.0,
                rfps=24.0)
        ]

    def test_activitynet_proposals2json(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        result_dict = activitynet_dataset.proposals2json(results)
        assert result_dict == dict(
            test1=[{
                'segment': [0.1, 0.9],
                'score': 0.1
            }],
            test2=[{
                'segment': [10.1, 20.9],
                'score': 0.9
            }])
        result_dict = activitynet_dataset.proposals2json(results, True)
        assert result_dict == dict(
            test1=[{
                'segment': [0.1, 0.9],
                'score': 0.1
            }],
            test2=[{
                'segment': [10.1, 20.9],
                'score': 0.9
            }])

    def test_activitynet_evaluate(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)

        with pytest.raises(TypeError):
            # results must be a list
            activitynet_dataset.evaluate('0.5')

        with pytest.raises(AssertionError):
            # The length of results must be equal to the dataset len
            activitynet_dataset.evaluate([0] * 5)

        with pytest.raises(KeyError):
            # unsupported metric
            activitynet_dataset.evaluate(
                [0] * len(activitynet_dataset), metrics='iou')

        # evaluate AR@AN metric
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        eval_result = activitynet_dataset.evaluate(results, metrics=['AR@AN'])
        assert set(eval_result) == set(
            ['auc', 'AR@1', 'AR@5', 'AR@10', 'AR@100'])

    def test_activitynet_dump_results(self):
        activitynet_dataset = ActivityNetDataset(self.action_ann_file,
                                                 self.action_pipeline,
                                                 self.data_prefix)
        # test dumping json file
        results = [
            dict(
                video_name='v_test1',
                proposal_list=[dict(segment=[0.1, 0.9], score=0.1)]),
            dict(
                video_name='v_test2',
                proposal_list=[dict(segment=[10.1, 20.9], score=0.9)])
        ]
        dump_results = {
            'version': 'VERSION 1.3',
            'results': {
                'test1': [{
                    'segment': [0.1, 0.9],
                    'score': 0.1
                }],
                'test2': [{
                    'segment': [10.1, 20.9],
                    'score': 0.9
                }]
            },
            'external_data': {}
        }

        tmp_filename = osp.join(tempfile.gettempdir(), 'result.json')
        activitynet_dataset.dump_results(results, tmp_filename, 'json')
        assert osp.isfile(tmp_filename)
        with open(tmp_filename, 'r+') as f:
            load_obj = mmcv.load(f, file_format='json')
        assert load_obj == dump_results
        os.remove(tmp_filename)

        # test dumping csv file
        results = [('test_video', np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9,
                                                              10]]))]
        with tempfile.TemporaryDirectory() as tmpdir:
            activitynet_dataset.dump_results(results, tmpdir, 'csv')
            load_obj = np.loadtxt(
                osp.join(tmpdir, 'test_video.csv'),
                dtype=np.float32,
                delimiter=',',
                skiprows=1)
            assert_array_equal(
                load_obj,
                np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                         dtype=np.float32))
