# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

import mmcv
import numpy as np
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_equal

from mmaction.datasets.transforms import (AudioFeatureSelector,
                                          DenseSampleFrames, SampleAVAFrames,
                                          SampleFrames, UntrimmedSampleFrames)


class BaseTestLoading:

    @classmethod
    def setup_class(cls):
        cls.data_prefix = osp.normpath(
            osp.join(osp.dirname(__file__), '../../data'))
        cls.img_path = osp.join(cls.data_prefix, 'test.jpg')
        cls.video_path = osp.join(cls.data_prefix, 'test.mp4')
        cls.wav_path = osp.join(cls.data_prefix, 'test.wav')
        cls.audio_spec_path = osp.join(cls.data_prefix, 'test.npy')
        cls.img_dir = osp.join(cls.data_prefix, 'imgs')
        cls.raw_feature_dir = osp.join(cls.data_prefix, 'activitynet_features')
        cls.bsp_feature_dir = osp.join(cls.data_prefix, 'bsp_features')
        cls.proposals_dir = osp.join(cls.data_prefix, 'proposals')

        cls.total_frames = 5
        cls.filename_tmpl = 'img_{:05}.jpg'
        cls.flow_filename_tmpl = '{}_{:05d}.jpg'
        video_total_frames = len(mmcv.VideoReader(cls.video_path))
        cls.audio_total_frames = video_total_frames

        cls.video_results = dict(
            filename=cls.video_path,
            label=1,
            total_frames=video_total_frames,
            start_index=0)
        cls.audio_results = dict(
            audios=np.random.randn(1280, ),
            audio_path=cls.wav_path,
            total_frames=cls.audio_total_frames,
            label=1,
            start_index=0)
        cls.audio_feature_results = dict(
            audios=np.random.randn(128, 80),
            audio_path=cls.audio_spec_path,
            total_frames=cls.audio_total_frames,
            label=1,
            start_index=0)
        cls.frame_results = dict(
            frame_dir=cls.img_dir,
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            start_index=1,
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
        """
        from mmaction.datasets.ssn_dataset import SSNInstance
        cls.proposal_results = dict(
            frame_dir=cls.img_dir,
            video_id='imgs',
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            start_index=1,
            out_proposals=[[['imgs', SSNInstance(1, 4, 10, 1, 1, 1)], 0],
                           [['imgs', SSNInstance(2, 5, 10, 2, 1, 1)], 0]])
        """

        cls.ava_results = dict(
            fps=30, timestamp=902, timestamp_start=840, shot_info=(0, 27000))

        cls.hvu_label_example1 = dict(
            categories=['action', 'object', 'scene', 'concept'],
            category_nums=[2, 5, 3, 2],
            label=dict(action=[0], object=[2, 3], scene=[0, 1]))
        cls.hvu_label_example2 = dict(
            categories=['action', 'object', 'scene', 'concept'],
            category_nums=[2, 5, 3, 2],
            label=dict(action=[1], scene=[1, 2], concept=[1]))


class TestSampling(BaseTestLoading):

    def test_sample_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        # Sample Frame with tail Frames
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3, frame_interval=1, num_clips=5, keep_tail_frames=True)
        sample_frames = SampleFrames(**config)
        sample_frames(video_result)
        sample_frames(frame_result)

        # Sample Frame with no temporal_jitter
        # clip_len=3, frame_interval=1, num_clips=5
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=3, frame_interval=1, num_clips=5, temporal_jitter=False)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 15
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 15
        assert np.max(sample_frames_results['frame_inds']) <= 5
        assert np.min(sample_frames_results['frame_inds']) >= 1
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={3}, '
                                       f'frame_interval={1}, '
                                       f'num_clips={5}, '
                                       f'temporal_jitter={False}, '
                                       f'twice_sample={False}, '
                                       f'out_of_bound_opt=loop, '
                                       f'test_mode={False})')

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
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={5}, '
                                       f'frame_interval={1}, '
                                       f'num_clips={5}, '
                                       f'temporal_jitter={False}, '
                                       f'twice_sample={False}, '
                                       f'out_of_bound_opt=repeat_last, '
                                       f'test_mode={False})')

        def check_monotonous(arr):
            length = arr.shape[0]
            for i in range(length - 1):
                if arr[i] > arr[i + 1]:
                    return False
            return True

        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 25
        frame_inds = sample_frames_results['frame_inds'].reshape([5, 5])
        for i in range(5):
            assert check_monotonous(frame_inds[i])

        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 25
        frame_inds = sample_frames_results['frame_inds'].reshape([5, 5])
        for i in range(5):
            assert check_monotonous(frame_inds[i])
        assert np.max(sample_frames_results['frame_inds']) <= 5
        assert np.min(sample_frames_results['frame_inds']) >= 1

        # Sample Frame with temporal_jitter
        # clip_len=4, frame_interval=2, num_clips=5
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4, frame_interval=2, num_clips=5, temporal_jitter=True)
        sample_frames = SampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 20
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 20
        assert np.max(sample_frames_results['frame_inds']) <= 5
        assert np.min(sample_frames_results['frame_inds']) >= 1
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={4}, '
                                       f'frame_interval={2}, '
                                       f'num_clips={5}, '
                                       f'temporal_jitter={True}, '
                                       f'twice_sample={False}, '
                                       f'out_of_bound_opt=loop, '
                                       f'test_mode={False})')

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 24
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 24
        assert np.max(sample_frames_results['frame_inds']) <= 5
        assert np.min(sample_frames_results['frame_inds']) >= 1
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={4}, '
                                       f'frame_interval={1}, '
                                       f'num_clips={6}, '
                                       f'temporal_jitter={False}, '
                                       f'twice_sample={False}, '
                                       f'out_of_bound_opt=loop, '
                                       f'test_mode={True})')

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 18
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 18
        assert np.max(sample_frames_results['frame_inds']) <= 5
        assert np.min(sample_frames_results['frame_inds']) >= 1

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 8
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([1, 2, 2, 3, 4, 5, 5, 6]))

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
        assert sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 8
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([1, 2, 2, 3, 4, 5, 5, 6]))

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
        assert sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(sample_frames_results, target_keys)
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
        assert sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 240
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 240
        assert np.max(sample_frames_results['frame_inds']) <= 30
        assert np.min(sample_frames_results['frame_inds']) >= 1

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert sample_frames_results['start_index'] == 0
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
        assert sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 24
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 24
        assert np.max(sample_frames_results['frame_inds']) <= 10
        assert np.min(sample_frames_results['frame_inds']) >= 1

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
        assert sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 48
        sample_frames_results = sample_frames(frame_result)
        assert len(sample_frames_results['frame_inds']) == 48
        assert np.max(sample_frames_results['frame_inds']) <= 40
        assert np.min(sample_frames_results['frame_inds']) >= 1

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
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 240
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 240
        assert repr(dense_sample_frames) == (
            f'{dense_sample_frames.__class__.__name__}('
            f'clip_len={4}, '
            f'frame_interval={1}, '
            f'num_clips={6}, '
            f'sample_range={64}, '
            f'num_sample_positions={10}, '
            f'temporal_jitter={False}, '
            f'out_of_bound_opt=loop, '
            f'test_mode={True})')

        # Dense sample with no temporal_jitter
        # clip_len=4, frame_interval=1, num_clips=6
        video_result = copy.deepcopy(self.video_results)
        frame_result = copy.deepcopy(self.frame_results)
        config = dict(
            clip_len=4, frame_interval=1, num_clips=6, temporal_jitter=False)
        dense_sample_frames = DenseSampleFrames(**config)
        dense_sample_frames_results = dense_sample_frames(video_result)
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
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
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
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
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 24
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 24
        assert repr(dense_sample_frames) == (
            f'{dense_sample_frames.__class__.__name__}('
            f'clip_len={4}, '
            f'frame_interval={1}, '
            f'num_clips={6}, '
            f'sample_range={32}, '
            f'num_sample_positions={10}, '
            f'temporal_jitter={False}, '
            f'out_of_bound_opt=loop, '
            f'test_mode={False})')

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
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
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
        assert dense_sample_frames_results['start_index'] == 0
        assert assert_dict_has_keys(dense_sample_frames_results, target_keys)
        assert len(dense_sample_frames_results['frame_inds']) == 120
        dense_sample_frames_results = dense_sample_frames(frame_result)
        assert len(dense_sample_frames_results['frame_inds']) == 120
        assert repr(dense_sample_frames) == (
            f'{dense_sample_frames.__class__.__name__}('
            f'clip_len={4}, '
            f'frame_interval={1}, '
            f'num_clips={6}, '
            f'sample_range={32}, '
            f'num_sample_positions={5}, '
            f'temporal_jitter={False}, '
            f'out_of_bound_opt=loop, '
            f'test_mode={True})')

    def test_untrim_sample_frames(self):

        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        frame_result = dict(
            frame_dir=None,
            total_frames=100,
            filename_tmpl=None,
            modality='RGB',
            start_index=0,
            label=1)
        video_result = copy.deepcopy(self.video_results)

        config = dict(clip_len=1, clip_interval=16)  # , start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(frame_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 6
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([8, 24, 40, 56, 72, 88]))
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'clip_interval={16}, '
                                       f'frame_interval={1})')

        config = dict(clip_len=1, clip_interval=16)  # , start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        frame_inds = np.array(list(range(8, 300, 16)))
        assert len(sample_frames_results['frame_inds']) == frame_inds.shape[0]
        assert_array_equal(sample_frames_results['frame_inds'], frame_inds)
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'clip_interval={16}, '
                                       f'frame_interval={1})')

        config = dict(clip_len=1, clip_interval=16)
        sample_frames = UntrimmedSampleFrames(**config)
        frame_result_ = copy.deepcopy(frame_result)
        frame_result_['start_index'] = 1
        sample_frames_results = sample_frames(frame_result_)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 6
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([8, 24, 40, 56, 72, 88]) + 1)
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'clip_interval={16}, '
                                       f'frame_interval={1})')

        config = dict(clip_len=3, clip_interval=16)  # , start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(frame_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 18
        assert_array_equal(
            sample_frames_results['frame_inds'],
            np.array([
                7, 8, 9, 23, 24, 25, 39, 40, 41, 55, 56, 57, 71, 72, 73, 87,
                88, 89
            ]))
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={3}, '
                                       f'clip_interval={16}, '
                                       f'frame_interval={1})')

        config = dict(
            clip_len=3, clip_interval=16, frame_interval=4)  # , start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(frame_result)
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 18
        assert_array_equal(
            sample_frames_results['frame_inds'],
            np.array([
                4, 8, 12, 20, 24, 28, 36, 40, 44, 52, 56, 60, 68, 72, 76, 84,
                88, 92
            ]))
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={3}, '
                                       f'clip_interval={16}, '
                                       f'frame_interval={4})')

    def test_sample_ava_frames(self):
        target_keys = [
            'fps', 'timestamp', 'timestamp_start', 'shot_info', 'frame_inds',
            'clip_len', 'frame_interval'
        ]
        config = dict(clip_len=32, frame_interval=2)
        sample_ava_dataset = SampleAVAFrames(**config)
        ava_result = sample_ava_dataset(results=self.ava_results)
        assert assert_dict_has_keys(ava_result, target_keys)
        assert ava_result['clip_len'] == 32
        assert ava_result['frame_interval'] == 2
        assert len(ava_result['frame_inds']) == 32
        assert repr(sample_ava_dataset) == (
            f'{sample_ava_dataset.__class__.__name__}('
            f'clip_len={32}, '
            f'frame_interval={2}, '
            f'test_mode={False})')

        # add test case in Issue #306
        config = dict(clip_len=8, frame_interval=8)
        sample_ava_dataset = SampleAVAFrames(**config)
        ava_result = sample_ava_dataset(results=self.ava_results)
        assert assert_dict_has_keys(ava_result, target_keys)
        assert ava_result['clip_len'] == 8
        assert ava_result['frame_interval'] == 8
        assert len(ava_result['frame_inds']) == 8
        assert repr(sample_ava_dataset) == (
            f'{sample_ava_dataset.__class__.__name__}('
            f'clip_len={8}, '
            f'frame_interval={8}, '
            f'test_mode={False})')

    """ TODO
    def test_sample_proposal_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames', 'start_index'
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
            sample_frames(proposal_result)

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={2}, '
                                       f'aug_segments={(1, 1)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=train)')

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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={2}, '
                                       f'aug_segments={(1, 1)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={True}, '
                                       f'mode=train)')

        # Sample Frame with no temporal_jitter in val mode
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposals)
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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={2}, '
                                       f'aug_segments={(1, 1)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=val)')

        # Sample Frame with no temporal_jitter in test mode
        # test_interval=2
        proposal_result = copy.deepcopy(self.proposals)
        proposal_result['out_proposals'] = None
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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 5
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={2}, '
                                       f'aug_segments={(1, 1)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={2}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=test)')

        # Sample Frame with no temporal_jitter to get clip_offsets zero
        # clip_len=1, frame_interval=1
        # body_segments=2, aug_segments=(1, 1)
        proposal_result = copy.deepcopy(self.proposals)
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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 8
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={2}, '
                                       f'aug_segments={(1, 1)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=train)')

        # Sample Frame with no temporal_jitter to
        # get clip_offsets zero in val mode
        # clip_len=1, frame_interval=1
        # body_segments=4, aug_segments=(2, 2)
        proposal_result = copy.deepcopy(self.proposals)
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
        assert assert_dict_has_keys(sample_frames_results, target_keys)
        assert len(sample_frames_results['frame_inds']) == 16
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={4}, '
                                       f'aug_segments={(2, 2)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=val)')
    """

    def test_audio_feature_selector(self):
        target_keys = ['audios']
        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.audio_feature_results)
        inputs['frame_inds'] = np.arange(0, self.audio_total_frames,
                                         2)[:, np.newaxis]
        inputs['num_clips'] = 1
        inputs['length'] = 1280
        audio_feature_selector = AudioFeatureSelector()
        results = audio_feature_selector(inputs)
        assert assert_dict_has_keys(results, target_keys)
        assert repr(audio_feature_selector) == (
            f'{audio_feature_selector.__class__.__name__}('
            f'fix_length={128})')
