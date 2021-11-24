# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets import AVADataset


class TestAVADataset:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), '../../data', 'ava_dataset'))
        cls.label_file = osp.join(cls.data_prefix, 'action_list.txt')
        cls.ann_file = osp.join(cls.data_prefix, 'ava_sample.csv')
        cls.exclude_file = osp.join(cls.data_prefix,
                                    'ava_excluded_timestamps_sample.csv')
        cls.proposal_file = osp.join(cls.data_prefix,
                                     'ava_proposals_sample.pkl')
        cls.pipeline = [
            dict(dict(type='SampleAVAFrames', clip_len=32, frame_interval=2))
        ]
        cls.proposal = mmcv.load(cls.proposal_file)

    def test_ava_dataset(self):
        target_keys = [
            'frame_dir', 'video_id', 'timestamp', 'img_key', 'shot_info',
            'fps', 'ann'
        ]
        ann_keys = ['gt_labels', 'gt_bboxes', 'entity_ids']
        pkl_keys = ['0f39OWEqJ24,0902', '0f39OWEqJ24,0903', '_-Z6wFjXtGQ,0902']

        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        ava_infos = ava_dataset.video_infos
        assert assert_dict_has_keys(ava_dataset.proposals, pkl_keys)

        assert assert_dict_has_keys(ava_infos[0], target_keys)
        assert assert_dict_has_keys(ava_infos[0]['ann'], ann_keys)
        assert len(ava_infos) == 1
        assert ava_infos[0]['frame_dir'] == osp.join(self.data_prefix,
                                                     '0f39OWEqJ24')
        assert ava_infos[0]['video_id'] == '0f39OWEqJ24'
        assert ava_infos[0]['timestamp'] == 902
        assert ava_infos[0]['img_key'] == '0f39OWEqJ24,0902'
        assert ava_infos[0]['shot_info'] == (0, 27000)
        assert ava_infos[0]['fps'] == 30
        assert len(ava_infos[0]['ann']) == 3
        target_labels = np.array([12, 17, 79])
        labels = np.zeros([81])
        labels[target_labels] = 1.
        target_labels = labels[None, ...]
        assert_array_equal(ava_infos[0]['ann']['gt_labels'], target_labels)
        assert_array_equal(ava_infos[0]['ann']['gt_bboxes'],
                           np.array([[0.031, 0.162, 0.67, 0.995]]))
        assert_array_equal(ava_infos[0]['ann']['entity_ids'], np.array([0]))

        # custom classes
        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            label_file=self.label_file,
            custom_classes=[17, 79],
            num_classes=3,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        ava_infos = ava_dataset.video_infos
        target_labels = np.array([1, 2])
        labels = np.zeros([3])
        labels[target_labels] = 1.
        target_labels = labels[None, ...]
        assert_array_equal(ava_infos[0]['ann']['gt_labels'], target_labels)
        assert_array_equal(ava_infos[0]['ann']['gt_bboxes'],
                           np.array([[0.031, 0.162, 0.67, 0.995]]))
        assert_array_equal(ava_infos[0]['ann']['entity_ids'], np.array([0]))

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        ava_infos = ava_dataset.video_infos
        assert len(ava_infos) == 3

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            test_mode=True,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        ava_infos = ava_dataset.video_infos
        assert len(ava_infos) == 3

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            test_mode=True,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)

    def test_ava_pipeline(self):
        target_keys = [
            'frame_dir', 'video_id', 'timestamp', 'img_key', 'shot_info',
            'fps', 'filename_tmpl', 'modality', 'start_index',
            'timestamp_start', 'timestamp_end', 'proposals', 'scores',
            'frame_inds', 'clip_len', 'frame_interval', 'gt_labels',
            'gt_bboxes', 'entity_ids'
        ]

        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        result = ava_dataset[0]
        assert assert_dict_has_keys(result, target_keys)

        assert result['filename_tmpl'] == 'img_{:05}.jpg'
        assert result['modality'] == 'RGB'
        assert result['start_index'] == 0
        assert result['timestamp_start'] == 900
        assert result['timestamp_end'] == 1800
        assert_array_equal(result['proposals'],
                           np.array([[0.011, 0.157, 0.655, 0.983]]))
        assert_array_equal(result['scores'], np.array([0.998163]))

        assert result['clip_len'] == 32
        assert result['frame_interval'] == 2
        assert len(result['frame_inds']) == 32

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            test_mode=True,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        # Try to get a sample
        result = ava_dataset[0]
        assert result['filename_tmpl'] == 'img_{:05}.jpg'
        assert result['modality'] == 'RGB'
        assert result['start_index'] == 0
        assert result['timestamp_start'] == 900
        assert result['timestamp_end'] == 1800

    @staticmethod
    def test_ava_evaluate():
        data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), '../../data', 'eval_detection'))
        ann_file = osp.join(data_prefix, 'gt.csv')
        label_file = osp.join(data_prefix, 'action_list.txt')

        ava_dataset = AVADataset(
            ann_file, None, [], label_file=label_file, num_classes=4)
        fake_result = [[
            np.array([[0.362, 0.156, 0.969, 0.666, 0.106],
                      [0.442, 0.083, 0.721, 0.947, 0.162]]),
            np.array([[0.288, 0.365, 0.766, 0.551, 0.706],
                      [0.178, 0.296, 0.707, 0.995, 0.223]]),
            np.array([[0.417, 0.167, 0.843, 0.939, 0.015],
                      [0.35, 0.421, 0.57, 0.689, 0.427]])
        ],
                       [
                           np.array([[0.256, 0.338, 0.726, 0.799, 0.563],
                                     [0.071, 0.256, 0.64, 0.75, 0.297]]),
                           np.array([[0.326, 0.036, 0.513, 0.991, 0.405],
                                     [0.351, 0.035, 0.729, 0.936, 0.945]]),
                           np.array([[0.051, 0.005, 0.975, 0.942, 0.424],
                                     [0.347, 0.05, 0.97, 0.944, 0.396]])
                       ],
                       [
                           np.array([[0.39, 0.087, 0.833, 0.616, 0.447],
                                     [0.461, 0.212, 0.627, 0.527, 0.036]]),
                           np.array([[0.022, 0.394, 0.93, 0.527, 0.109],
                                     [0.208, 0.462, 0.874, 0.948, 0.954]]),
                           np.array([[0.206, 0.456, 0.564, 0.725, 0.685],
                                     [0.106, 0.445, 0.782, 0.673, 0.367]])
                       ]]
        res = ava_dataset.evaluate(fake_result)
        assert_array_almost_equal(res['mAP@0.5IOU'], 0.027777778)

        # custom classes
        ava_dataset = AVADataset(
            ann_file,
            None, [],
            label_file=label_file,
            num_classes=3,
            custom_classes=[1, 3])
        fake_result = [[
            np.array([[0.362, 0.156, 0.969, 0.666, 0.106],
                      [0.442, 0.083, 0.721, 0.947, 0.162]]),
            np.array([[0.417, 0.167, 0.843, 0.939, 0.015],
                      [0.35, 0.421, 0.57, 0.689, 0.427]])
        ],
                       [
                           np.array([[0.256, 0.338, 0.726, 0.799, 0.563],
                                     [0.071, 0.256, 0.64, 0.75, 0.297]]),
                           np.array([[0.051, 0.005, 0.975, 0.942, 0.424],
                                     [0.347, 0.05, 0.97, 0.944, 0.396]])
                       ],
                       [
                           np.array([[0.39, 0.087, 0.833, 0.616, 0.447],
                                     [0.461, 0.212, 0.627, 0.527, 0.036]]),
                           np.array([[0.206, 0.456, 0.564, 0.725, 0.685],
                                     [0.106, 0.445, 0.782, 0.673, 0.367]])
                       ]]
        res = ava_dataset.evaluate(fake_result)
        assert_array_almost_equal(res['mAP@0.5IOU'], 0.04166667)
