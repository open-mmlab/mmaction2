# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine
import numpy as np
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets import AVADataset, AVAKineticsDataset
from mmaction.utils import register_all_modules


class TestAVADataset:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), './../data', 'ava_dataset'))
        cls.label_file = osp.join(cls.data_prefix, 'action_list.txt')
        cls.ann_file = osp.join(cls.data_prefix, 'ava_sample.csv')
        cls.exclude_file = osp.join(cls.data_prefix,
                                    'ava_excluded_timestamps_sample.csv')
        cls.proposal_file = osp.join(cls.data_prefix,
                                     'ava_proposals_sample.pkl')
        cls.pipeline = [
            dict(dict(type='SampleAVAFrames', clip_len=32, frame_interval=2))
        ]
        cls.proposal = mmengine.load(cls.proposal_file)

    def test_ava_dataset(self):
        register_all_modules()
        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            self.label_file,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        # custom classes
        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            label_file=self.label_file,
            custom_classes=[17, 79],
            num_classes=3,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)
        # ava_infos = ava_dataset.video_infos
        target_labels = np.array([1, 2])
        labels = np.zeros([3])
        labels[target_labels] = 1.
        target_labels = labels[None, ...]

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            self.label_file,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        ava_dataset = AVADataset(
            self.ann_file,
            None,
            self.pipeline,
            self.label_file,
            test_mode=True,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        del ava_dataset

    def test_ava_pipeline(self):
        register_all_modules()
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
            self.label_file,
            data_prefix={'img': self.data_prefix},
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
            self.label_file,
            test_mode=True,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)
        # Try to get a sample
        result = ava_dataset[0]
        assert result['filename_tmpl'] == 'img_{:05}.jpg'
        assert result['modality'] == 'RGB'
        assert result['start_index'] == 0
        assert result['timestamp_start'] == 900
        assert result['timestamp_end'] == 1800


class TestAVAKineticsDataset:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), './../data', 'ava_dataset'))
        cls.label_file = osp.join(cls.data_prefix, 'action_list.txt')
        cls.ann_file = osp.join(cls.data_prefix, 'ava_sample.csv')
        cls.exclude_file = osp.join(cls.data_prefix,
                                    'ava_excluded_timestamps_sample.csv')
        cls.proposal_file = osp.join(cls.data_prefix,
                                     'ava_proposals_sample.pkl')
        cls.pipeline = [
            dict(dict(type='SampleAVAFrames', clip_len=32, frame_interval=2))
        ]
        cls.proposal = mmengine.load(cls.proposal_file)

    def test_ava_kinetics_dataset(self):
        register_all_modules()
        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            self.label_file,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        # custom classes
        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            label_file=self.label_file,
            custom_classes=[17, 79],
            num_classes=3,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)
        # ava_infos = ava_dataset.video_infos
        target_labels = np.array([1, 2])
        labels = np.zeros([3])
        labels[target_labels] = 1.
        target_labels = labels[None, ...]

        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            None,
            self.pipeline,
            self.label_file,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            None,
            self.pipeline,
            self.label_file,
            test_mode=True,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)

        del ava_dataset

    def test_ava_kinetics_pipeline(self):
        register_all_modules()
        target_keys = [
            'frame_dir', 'video_id', 'timestamp', 'img_key', 'shot_info',
            'fps', 'filename_tmpl', 'modality', 'start_index',
            'timestamp_start', 'timestamp_end', 'proposals', 'scores',
            'frame_inds', 'clip_len', 'frame_interval', 'gt_labels',
            'gt_bboxes', 'entity_ids'
        ]

        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            self.label_file,
            data_prefix={'img': self.data_prefix},
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

        ava_dataset = AVAKineticsDataset(
            self.ann_file,
            None,
            self.pipeline,
            self.label_file,
            test_mode=True,
            data_prefix={'img': self.data_prefix},
            proposal_file=self.proposal_file)
        # Try to get a sample
        result = ava_dataset[0]
        assert result['filename_tmpl'] == 'img_{:05}.jpg'
        assert result['modality'] == 'RGB'
        assert result['start_index'] >= 0
        assert result['timestamp_start'] > 0
        assert result['timestamp_end'] > result['timestamp_start']
