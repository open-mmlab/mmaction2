import os.path as osp

import mmcv
import numpy as np
from numpy.testing import assert_array_equal

from mmaction.datasets import AVADataset


def check_keys_contain(result_keys, target_keys):
    """Check if all elements in target_keys is in result_keys."""
    return set(target_keys).issubset(set(result_keys))


class TestAVADataset(object):

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data', 'test_ava_dataset')
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
        ann_keys = ['labels', 'entity_boxes', 'entity_ids']
        pkl_keys = ['0f39OWEqJ24,0902', '0f39OWEqJ24,0903', '_-Z6wFjXtGQ,0902']

        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        ava_infos = ava_dataset.video_infos
        assert check_keys_contain(ava_dataset.proposals.keys(), pkl_keys)

        assert check_keys_contain(ava_infos[0].keys(), target_keys)
        assert check_keys_contain(ava_infos[0]['ann'].keys(), ann_keys)
        assert len(ava_infos) == 1
        assert ava_infos[0]['frame_dir'] == osp.join(self.data_prefix,
                                                     '0f39OWEqJ24')
        assert ava_infos[0]['video_id'] == '0f39OWEqJ24'
        assert ava_infos[0]['timestamp'] == 902
        assert ava_infos[0]['img_key'] == '0f39OWEqJ24,0902'
        assert ava_infos[0]['shot_info'] == (0, 26880)
        assert ava_infos[0]['fps'] == 30
        assert len(ava_infos[0]['ann']) == 3
        target_labels = np.array([12, 17, 79] + [
            -1,
        ] * 78)
        target_labels = target_labels[None, ...]
        assert_array_equal(ava_infos[0]['ann']['labels'], target_labels)
        assert_array_equal(ava_infos[0]['ann']['entity_boxes'],
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
            proposal_file=None)
        assert ava_dataset.proposals is None

    def test_ava_pipeline(self):
        target_keys = [
            'frame_dir', 'video_id', 'timestamp', 'img_key', 'shot_info',
            'fps', 'ann', 'filename_tmpl', 'modality', 'start_index',
            'timestamp_start', 'timestamp_end', 'proposals', 'frame_inds',
            'clip_len', 'frame_interval'
        ]
        ann_keys = ['labels', 'entity_boxes', 'entity_ids']

        ava_dataset = AVADataset(
            self.ann_file,
            self.exclude_file,
            self.pipeline,
            data_prefix=self.data_prefix,
            proposal_file=self.proposal_file)
        result = ava_dataset[0]
        assert check_keys_contain(result.keys(), target_keys)
        assert check_keys_contain(result['ann'].keys(), ann_keys)

        assert result['filename_tmpl'] == 'img_{:05}.jpg'
        assert result['modality'] == 'RGB'
        assert result['start_index'] == 1
        assert result['timestamp_start'] == 902
        assert result['timestamp_end'] == 1798
        assert_array_equal(result['proposals'],
                           np.array([[0.011, 0.157, 0.655, 0.983, 0.998163]]))

        assert result['clip_len'] == 32
        assert result['frame_interval'] == 2
        assert len(result['frame_inds']) == 32
