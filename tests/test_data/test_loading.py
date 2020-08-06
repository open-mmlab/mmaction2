import copy
import os.path as osp

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.pipelines import (DecordDecode, DecordInit,
                                         DenseSampleFrames, FrameSelector,
                                         GenerateLocalizationLabels,
                                         LoadLocalizationFeature,
                                         LoadProposals, OpenCVDecode,
                                         OpenCVInit, PyAVDecode, PyAVInit,
                                         SampleFrames, SampleProposalFrames)


class ExampleSSNInstance(object):

    def __init__(self,
                 start_frame,
                 end_frame,
                 num_frames,
                 label=None,
                 best_iou=None,
                 overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, num_frames)
        self.label = label if label is not None else -1
        self.coverage = (end_frame - start_frame) / num_frames
        self.best_iou = best_iou
        self.overlap_self = overlap_self


class TestLoading(object):

    @staticmethod
    def check_keys_contain(result_keys, target_keys):
        """Check if all elements in target_keys is in result_keys."""
        return set(target_keys).issubset(set(result_keys))

    @classmethod
    def setup_class(cls):
        cls.img_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.jpg')
        cls.video_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.mp4')
        cls.img_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_imgs')
        cls.raw_feature_dir = osp.join(
            osp.dirname(osp.dirname(__file__)),
            'data/test_activitynet_features')
        cls.bsp_feature_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_bsp_features')
        cls.proposals_dir = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test_proposals')
        cls.total_frames = 5
        cls.filename_tmpl = 'img_{:05}.jpg'
        cls.flow_filename_tmpl = '{}_{:05d}.jpg'
        cls.video_results = dict(filename=cls.video_path, label=1)
        cls.frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            modality='RGB',
            offset=0,
            label=1)
        cls.flow_frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.flow_filename_tmpl,
            modality='Flow',
            offset=0,
            label=1)
        cls.action_results = dict(
            video_name='v_test1',
            data_prefix=cls.raw_feature_dir,
            temporal_scale=5,
            boundary_ratio=0.1,
            duration_second=10,
            duration_frame=10,
            feature_frame=8,
            annotations=[{
                'segment': [3.0, 5.0],
                'label': 'Rock climbing'
            }])
        cls.proposal_results = dict(
            frame_dir=cls.img_dir,
            video_id='test_imgs',
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            out_props=[[['test_imgs',
                         ExampleSSNInstance(1, 4, 10, 1, 1, 1)], 0],
                       [['test_imgs',
                         ExampleSSNInstance(2, 5, 10, 2, 1, 1)], 0]])

    def test_sample_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        # Sample Frame with no temporal_jitter
        # clip_len=3, frame_interval=1, num_clips=5
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3, frame_interval=1, num_clips=5, temporal_jitter=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 15
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 15

        # Sample Frame with no temporal_jitter
        # clip_len=5, frame_interval=1, num_clips=5,
        # out_of_bound_opt='repeat_last'
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=5,
            frame_interval=1,
            num_clips=5,
            temporal_jitter=False,
            out_of_bound_opt='repeat_last')
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)

        def check_monotonous(arr):
            length = arr.shape[0]
            for i in range(length - 1):
                if arr[i] > arr[i + 1]:
                    return False
            return True

        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 25
        frame_inds = sample_frames_results['frame_inds'].reshape([5, 5])
        for i in range(5):
            assert check_monotonous(frame_inds[i])

        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 25
        frame_inds = sample_frames_results['frame_inds'].reshape([5, 5])
        for i in range(5):
            assert check_monotonous(frame_inds[i])

        # Sample Frame with temporal_jitter
        # clip_len=4, frame_interval=2, num_clips=5
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4, frame_interval=2, num_clips=5, temporal_jitter=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 20
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 20

        # Sample Frame with no temporal_jitter in test mode
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            temporal_jitter=False,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 24
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 24

        # Sample Frame with no temporal_jitter in test mode
        # clip_len=3, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3,
            frame_interval=1,
            num_clips=6,
            temporal_jitter=False,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 18
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 18

        # Sample Frame with no temporal_jitter to get clip_offsets
        # clip_len=1, frame_interval=1, num_clips=8
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 6
        config = dict(
            clip_len=1,
            frame_interval=1,
            num_clips=8,
            temporal_jitter=False,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 8
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([1, 2, 2, 3, 4, 5, 5, 6]))

        # Sample Frame with no temporal_jitter to get clip_offsets
        # clip_len=1, frame_interval=1, num_clips=8, start_index=0
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 6
        config = dict(
            clip_len=1,
            frame_interval=1,
            num_clips=8,
            start_index=0,
            temporal_jitter=False,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 8
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([0, 1, 1, 2, 3, 4, 4, 5]))

        # Sample Frame with no temporal_jitter to get clip_offsets zero
        # clip_len=6, frame_interval=1, num_clips=1
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 5
        config = dict(
            clip_len=6,
            frame_interval=1,
            num_clips=1,
            temporal_jitter=False,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 6
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 6
        assert_array_equal(sample_frames_results['frame_inds'],
                           [1, 2, 3, 4, 5, 1])

        # Sample Frame with no temporal_jitter to get avg_interval <= 0
        # clip_len=12, frame_interval=1, num_clips=20
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 30
        config = dict(
            clip_len=12,
            frame_interval=1,
            num_clips=20,
            temporal_jitter=False,
            test_mode=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 240
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 240

        # Sample Frame with no temporal_jitter to get clip_offsets
        # clip_len=1, frame_interval=1, num_clips=8
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 6
        config = dict(
            clip_len=1,
            frame_interval=1,
            num_clips=8,
            temporal_jitter=False,
            test_mode=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 8
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([1, 2, 3, 3, 4, 5, 5, 6]))

        # Sample Frame with no temporal_jitter to get clip_offsets zero
        # clip_len=12, frame_interval=1, num_clips=2
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 10
        config = dict(
            clip_len=12,
            frame_interval=1,
            num_clips=2,
            temporal_jitter=False,
            test_mode=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 24
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 24

        # Sample Frame using twice sample
        # clip_len=12, frame_interval=1, num_clips=2
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        frame_result['total_frames'] = 40
        config = dict(
            clip_len=12,
            frame_interval=1,
            num_clips=2,
            temporal_jitter=False,
            twice_sample=True,
            test_mode=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 48
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 48

    def test_dense_sample_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        # Dense sample with no temporal_jitter in test mode
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            temporal_jitter=False,
            test_mode=True)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 240
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 240

        # Dense sample with no temporal_jitter
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4, frame_interval=1, num_clips=6, temporal_jitter=False)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 24
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 24

        # Dense sample with no temporal_jitter, sample_range=32 in test mode
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            sample_range=32,
            temporal_jitter=False,
            test_mode=True)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 240
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 240

        # Dense sample with no temporal_jitter, sample_range=32
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            sample_range=32,
            temporal_jitter=False)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 24
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 24

        # Dense sample with no temporal_jitter, sample_range=1000 to check mod
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            sample_range=1000,
            temporal_jitter=False)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 24
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 24

        # Dense sample with no temporal_jitter in test mode
        # sample_range=32, num_sample_positions=5
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4,
            frame_interval=1,
            num_clips=6,
            num_sample_positions=5,
            sample_range=32,
            temporal_jitter=False,
            test_mode=True)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 120
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 120

    def test_sample_proposal_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        # test error cases
        with pytest.raises(TypeError):
            proposal_result = copy.deepcopy(self.proposal_results)
            config = dict(
                clip_len=1,
                frame_interval=1,
                body_segments=2,
                aug_segments=('error', 'error'),
                aug_ratio=0.5,
                temporal_jitter=False)
            sample_frames = SampleProposalFrames(**config)
            sample_frames_results = sample_frames(proposal_result)

        # test normal cases
        # Sample Frame with no temporal_jitter
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['total_frames'] = 9
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=2,
            aug_segments=(1, 1),
            aug_ratio=0.5,
            temporal_jitter=False)
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8

        # Sample Frame with temporal_jitter
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['total_frames'] = 9
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=2,
            aug_segments=(1, 1),
            aug_ratio=0.5,
            temporal_jitter=True)
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8

        # Sample Frame with no temporal_jitter in val mode
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['total_frames'] = 9
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=2,
            aug_segments=(1, 1),
            aug_ratio=0.5,
            temporal_jitter=False,
            mode='val')
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8

        # Sample Frame with no temporal_jitter in test mode
        # test_interval=2
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['out_props'] = None
        proposal_result['total_frames'] = 10
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=2,
            aug_segments=(1, 1),
            aug_ratio=0.5,
            test_interval=2,
            temporal_jitter=False,
            mode='test')
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 5

        # Sample Frame with no temporal_jitter to get clip_offsets zero
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['total_frames'] = 3
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=2,
            aug_segments=(1, 1),
            aug_ratio=0.5,
            temporal_jitter=False)
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 8

        # Sample Frame with no temporal_jitter to
        # get clip_offsets zero in val mode
        # clip_len=1, frame_interval=1
        # body_segments=4, aug_segments=(2, 2)
        proposal_result = copy.deepcopy(self.proposal_results)
        proposal_result['total_frames'] = 3
        config = dict(
            clip_len=1,
            frame_interval=1,
            body_segments=4,
            aug_segments=(2, 2),
            aug_ratio=0.5,
            temporal_jitter=False,
            mode='val')
        sample_frames = SampleProposalFrames(**config)
        sample_frames_results = sample_frames(proposal_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 16

    def test_pyav_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        assert self.check_keys_contain(pyav_init_result.keys(), target_keys)
        assert pyav_init_result['total_frames'] == 300

    def test_pyav_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test PyAV with 2 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               2)[:, np.newaxis]
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test PyAV with 1 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # PyAV with multi thread and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode(multi_thread=True)
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test PyAV with 2 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames,
                                               2)[:, np.newaxis]
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test PyAV with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode()
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # PyAV with multi thread
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 5)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        video_result['video_reader'] = pyav_init_result['video_reader']

        pyav_decode = PyAVDecode(multi_thread=True)
        pyav_decode_result = pyav_decode(video_result)
        assert self.check_keys_contain(pyav_decode_result.keys(), target_keys)
        assert pyav_decode_result['original_shape'] == (256, 340)
        assert np.shape(pyav_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        assert repr(pyav_decode) == pyav_decode.__class__.__name__ + \
            f'(multi_thread={True})'

    def test_decord_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        assert self.check_keys_contain(decord_init_result.keys(), target_keys)
        assert decord_init_result['total_frames'] == len(
            decord_init_result['video_reader'])

    def test_decord_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test Decord with 2 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               3)[:, np.newaxis]
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 1 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 3)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 2 dim input and start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               3)[:, np.newaxis]
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test Decord with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 3)
        decord_init = DecordInit()
        decord_init_result = decord_init(video_result)
        video_result['video_reader'] = decord_init_result['video_reader']

        decord_decode = DecordDecode()
        decord_decode_result = decord_decode(video_result)
        assert self.check_keys_contain(decord_decode_result.keys(),
                                       target_keys)
        assert decord_decode_result['original_shape'] == (256, 340)
        assert np.shape(decord_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

    def test_opencv_init(self):
        target_keys = ['new_path', 'video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        assert self.check_keys_contain(opencv_init_result.keys(), target_keys)
        assert opencv_init_result['total_frames'] == len(
            opencv_init_result['video_reader'])

    def test_opencv_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape']

        # test OpenCV with 2 dim input when start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames,
                                               2)[:, np.newaxis]
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test OpenCV with 2 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames,
                                               2)[:, np.newaxis]
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

        # test OpenCV with 1 dim input when start_index = 0
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(0, self.total_frames, 3)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        # test OpenCV with 1 dim input
        video_result = copy.deepcopy(self.video_results)
        video_result['frame_inds'] = np.arange(1, self.total_frames, 3)
        opencv_init = OpenCVInit()
        opencv_init_result = opencv_init(video_result)
        video_result['video_reader'] = opencv_init_result['video_reader']

        opencv_decode = OpenCVDecode()
        opencv_decode_result = opencv_decode(video_result)
        assert self.check_keys_contain(opencv_decode_result.keys(),
                                       target_keys)
        assert opencv_decode_result['original_shape'] == (256, 340)
        assert np.shape(opencv_decode_result['imgs']) == (len(
            video_result['frame_inds']), 256, 340, 3)

    def test_frame_selector(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape', 'modality']

        # test frame selector with 2 dim input when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)[:,
                                                                  np.newaxis]
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)[:,
                                                                  np.newaxis]
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 5)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        # when start_index = 0
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = FrameSelector(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decording backend
        # when start_index = 0
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 5)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = FrameSelector(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decording backend
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = FrameSelector(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

    def test_load_localization_feature(self):
        target_keys = ['raw_feature']

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_localization_feature = LoadLocalizationFeature(
                'unsupport_ext')

        # test normal cases
        load_localization_feature = LoadLocalizationFeature()
        load_localization_feature_result = load_localization_feature(
            action_result)
        assert self.check_keys_contain(load_localization_feature_result.keys(),
                                       target_keys)
        assert load_localization_feature_result['raw_feature'].shape == (400,
                                                                         5)

    def test_generate_localization_label(self):
        action_result = copy.deepcopy(self.action_results)
        action_result['raw_feature'] = np.random.randn(400, 5)

        # test default setting
        target_keys = ['gt_bbox']
        generate_localization_labels = GenerateLocalizationLabels()
        generate_localization_labels_result = generate_localization_labels(
            action_result)
        assert self.check_keys_contain(
            generate_localization_labels_result.keys(), target_keys)

        assert_array_almost_equal(
            generate_localization_labels_result['gt_bbox'], [[0.375, 0.625]],
            decimal=4)

    def test_load_proposals(self):
        target_keys = [
            'bsp_feature', 'tmin', 'tmax', 'tmin_score', 'tmax_score',
            'reference_temporal_iou'
        ]

        action_result = copy.deepcopy(self.action_results)

        # test error cases
        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir,
                                           'unsupport_ext')

        with pytest.raises(NotImplementedError):
            load_proposals = LoadProposals(5, self.proposals_dir,
                                           self.bsp_feature_dir, '.csv',
                                           'unsupport_ext')

        # test normal cases
        load_proposals = LoadProposals(5, self.proposals_dir,
                                       self.bsp_feature_dir)
        load_proposals_result = load_proposals(action_result)
        assert self.check_keys_contain(load_proposals_result.keys(),
                                       target_keys)
        assert (load_proposals_result['bsp_feature'].shape[0] == 5)
        assert load_proposals_result['tmin'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin'], np.arange(0.1, 0.6, 0.1), decimal=4)
        assert load_proposals_result['tmax'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax'], np.arange(0.2, 0.7, 0.1), decimal=4)
        assert load_proposals_result['tmin_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmin_score'],
            np.arange(0.95, 0.90, -0.01),
            decimal=4)
        assert load_proposals_result['tmax_score'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['tmax_score'],
            np.arange(0.96, 0.91, -0.01),
            decimal=4)
        assert load_proposals_result['reference_temporal_iou'].shape == (5, )
        assert_array_almost_equal(
            load_proposals_result['reference_temporal_iou'],
            np.arange(0.85, 0.80, -0.01),
            decimal=4)
