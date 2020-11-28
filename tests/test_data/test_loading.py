import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
import torch
from numpy.testing import assert_array_almost_equal, assert_array_equal

# yapf: disable
from mmaction.datasets.pipelines import (AudioDecode, AudioDecodeInit,
                                         AudioFeatureSelector, DecordDecode,
                                         DecordInit, DenseSampleFrames,
                                         FrameSelector,
                                         GenerateLocalizationLabels,
                                         LoadAudioFeature, LoadHVULabel,
                                         LoadLocalizationFeature,
                                         LoadProposals, OpenCVDecode,
                                         OpenCVInit, PyAVDecode,
                                         PyAVDecodeMotionVector, PyAVInit,
                                         RawFrameDecode, SampleAVAFrames,
                                         SampleFrames, SampleProposalFrames,
                                         UntrimmedSampleFrames)

# yapf: enable


class ExampleSSNInstance:

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


class TestLoading:

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
        cls.wav_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.wav')
        cls.audio_spec_path = osp.join(
            osp.dirname(osp.dirname(__file__)), 'data/test.npy')
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
        cls.proposal_results = dict(
            frame_dir=cls.img_dir,
            video_id='test_imgs',
            total_frames=cls.total_frames,
            filename_tmpl=cls.filename_tmpl,
            start_index=1,
            out_proposals=[[[
                'test_imgs',
                ExampleSSNInstance(1, 4, 10, 1, 1, 1)
            ], 0], [['test_imgs',
                     ExampleSSNInstance(2, 5, 10, 2, 1, 1)], 0]])

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

    def test_load_hvu_label(self):
        hvu_label_example1 = copy.deepcopy(self.hvu_label_example1)
        hvu_label_example2 = copy.deepcopy(self.hvu_label_example2)
        categories = hvu_label_example1['categories']
        category_nums = hvu_label_example1['category_nums']
        num_tags = sum(category_nums)
        num_categories = len(categories)

        loader = LoadHVULabel()
        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={False})')

        result1 = loader(hvu_label_example1)
        label1 = torch.zeros(num_tags)
        mask1 = torch.zeros(num_tags)
        category_mask1 = torch.zeros(num_categories)

        assert repr(loader) == (f'{loader.__class__.__name__}('
                                f'hvu_initialized={True})')

        label1[[0, 4, 5, 7, 8]] = 1.
        mask1[:10] = 1.
        category_mask1[:3] = 1.

        assert torch.all(torch.eq(label1, result1['label']))
        assert torch.all(torch.eq(mask1, result1['mask']))
        assert torch.all(torch.eq(category_mask1, result1['category_mask']))

        result2 = loader(hvu_label_example2)
        label2 = torch.zeros(num_tags)
        mask2 = torch.zeros(num_tags)
        category_mask2 = torch.zeros(num_categories)

        label2[[1, 8, 9, 11]] = 1.
        mask2[:2] = 1.
        mask2[7:] = 1.
        category_mask2[[0, 2, 3]] = 1.

        assert torch.all(torch.eq(label2, result2['label']))
        assert torch.all(torch.eq(mask2, result2['mask']))
        assert torch.all(torch.eq(category_mask2, result2['category_mask']))

    def test_sample_frames(self):
        target_keys = [
            'frame_inds', 'clip_len', 'frame_interval', 'num_clips',
            'total_frames'
        ]

        with pytest.warns(UserWarning):
            # start_index has been deprecated
            config = dict(
                clip_len=3, frame_interval=1, num_clips=5, start_index=1)
            SampleFrames(**config)

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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert sample_frames_results['start_index'] == 0
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
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
        assert dense_sample_frames_results['start_index'] == 0
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
        assert dense_sample_frames_results['start_index'] == 0
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
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
        assert dense_sample_frames_results['start_index'] == 0
        assert self.check_keys_contain(dense_sample_frames_results.keys(),
                                       target_keys)
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

        config = dict(clip_len=1, frame_interval=16, start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(frame_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 6
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([8, 24, 40, 56, 72, 88]))
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'frame_interval={16})')

        config = dict(clip_len=1, frame_interval=16, start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(video_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        frame_inds = np.array(list(range(8, 300, 16)))
        assert len(sample_frames_results['frame_inds']) == frame_inds.shape[0]
        assert_array_equal(sample_frames_results['frame_inds'], frame_inds)
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'frame_interval={16})')

        config = dict(clip_len=1, frame_interval=16)
        sample_frames = UntrimmedSampleFrames(**config)
        frame_result_ = copy.deepcopy(frame_result)
        frame_result_['start_index'] = 1
        sample_frames_results = sample_frames(frame_result_)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 6
        assert_array_equal(sample_frames_results['frame_inds'],
                           np.array([8, 24, 40, 56, 72, 88]) + 1)
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'frame_interval={16})')

        config = dict(clip_len=3, frame_interval=16, start_index=0)
        sample_frames = UntrimmedSampleFrames(**config)
        sample_frames_results = sample_frames(frame_result)
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
        assert len(sample_frames_results['frame_inds']) == 18
        assert_array_equal(
            sample_frames_results['frame_inds'],
            np.array([
                7, 8, 9, 23, 24, 25, 39, 40, 41, 55, 56, 57, 71, 72, 73, 87,
                88, 89
            ]))
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={3}, '
                                       f'frame_interval={16})')

    def test_sample_ava_frames(self):
        target_keys = [
            'fps', 'timestamp', 'timestamp_start', 'shot_info', 'frame_inds',
            'clip_len', 'frame_interval'
        ]
        config = dict(clip_len=32, frame_interval=2)
        sample_ava_dataset = SampleAVAFrames(**config)
        ava_result = sample_ava_dataset(results=self.ava_results)
        assert self.check_keys_contain(ava_result.keys(), target_keys)
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
        assert self.check_keys_contain(ava_result.keys(), target_keys)
        assert ava_result['clip_len'] == 8
        assert ava_result['frame_interval'] == 8
        assert len(ava_result['frame_inds']) == 8
        assert repr(sample_ava_dataset) == (
            f'{sample_ava_dataset.__class__.__name__}('
            f'clip_len={8}, '
            f'frame_interval={8}, '
            f'test_mode={False})')

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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        proposal_result = copy.deepcopy(self.proposal_results)
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
        assert self.check_keys_contain(sample_frames_results.keys(),
                                       target_keys)
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
        assert repr(sample_frames) == (f'{sample_frames.__class__.__name__}('
                                       f'clip_len={1}, '
                                       f'body_segments={4}, '
                                       f'aug_segments={(2, 2)}, '
                                       f'aug_ratio={(0.5, 0.5)}, '
                                       f'frame_interval={1}, '
                                       f'test_interval={6}, '
                                       f'temporal_jitter={False}, '
                                       f'mode=val)')

    def test_pyav_init(self):
        target_keys = ['video_reader', 'total_frames']
        video_result = copy.deepcopy(self.video_results)
        pyav_init = PyAVInit()
        pyav_init_result = pyav_init(video_result)
        assert self.check_keys_contain(pyav_init_result.keys(), target_keys)
        assert pyav_init_result['total_frames'] == 300
        assert repr(
            pyav_init) == f'{pyav_init.__class__.__name__}(io_backend=disk)'

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
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={False})')

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
        assert repr(pyav_decode) == (f'{pyav_decode.__class__.__name__}('
                                     f'multi_thread={True})')

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
        assert repr(decord_init) == (f'{decord_init.__class__.__name__}('
                                     f'io_backend=disk, '
                                     f'num_threads={1})')

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
        assert repr(opencv_init) == (f'{opencv_init.__class__.__name__}('
                                     f'io_backend=disk)')

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

    def test_rawframe_selector(self):

        with pytest.warns(UserWarning):
            FrameSelector(io_backend='disk')

    def test_rawframe_decode(self):
        target_keys = ['frame_inds', 'imgs', 'original_shape', 'modality']

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)[:,
                                                                  np.newaxis]
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 2 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)[:,
                                                                  np.newaxis]
        frame_selector = RawFrameDecode(io_backend='disk')
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
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(0, self.total_frames, 2)
        # since the test images start with index 1, we plus 1 to frame_inds
        # in order to pass the CI
        inputs['frame_inds'] = inputs['frame_inds'] + 1
        frame_selector = RawFrameDecode(io_backend='disk')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']) * 2,
                                             240, 320)
        assert results['original_shape'] == (240, 320)

        # test frame selector with 1 dim input for flow images
        inputs = copy.deepcopy(self.flow_frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 2)
        frame_selector = RawFrameDecode(io_backend='disk')
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
        frame_selector = RawFrameDecode(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)

        # test frame selector in turbojpeg decording backend
        inputs = copy.deepcopy(self.frame_results)
        inputs['frame_inds'] = np.arange(1, self.total_frames, 5)
        frame_selector = RawFrameDecode(
            io_backend='disk', decoding_backend='turbojpeg')
        results = frame_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert np.shape(results['imgs']) == (len(inputs['frame_inds']), 240,
                                             320, 3)
        assert results['original_shape'] == (240, 320)
        assert repr(frame_selector) == (f'{frame_selector.__class__.__name__}('
                                        f'io_backend=disk, '
                                        f'decoding_backend=turbojpeg)')

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
        assert repr(load_localization_feature) == (
            f'{load_localization_feature.__class__.__name__}('
            f'raw_feature_ext=.csv)')

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
        assert repr(load_proposals) == (
            f'{load_proposals.__class__.__name__}('
            f'top_k={5}, '
            f'pgm_proposals_dir={self.proposals_dir}, '
            f'pgm_features_dir={self.bsp_feature_dir}, '
            f'proposal_ext=.csv, '
            f'feature_ext=.npy)')

    def test_audio_decode_init(self):
        target_keys = ['audios', 'length', 'sample_rate']
        inputs = copy.deepcopy(self.audio_results)
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)

        # test when no audio file exists
        inputs = copy.deepcopy(self.audio_results)
        inputs['audio_path'] = 'foo/foo/bar.wav'
        audio_decode_init = AudioDecodeInit()
        results = audio_decode_init(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert results['audios'].shape == (10.0 *
                                           audio_decode_init.sample_rate, )
        assert repr(audio_decode_init) == (
            f'{audio_decode_init.__class__.__name__}('
            f'io_backend=disk, '
            f'sample_rate=16000, '
            f'pad_method=zero)')

    def test_audio_decode(self):
        target_keys = ['frame_inds', 'audios']
        inputs = copy.deepcopy(self.audio_results)
        inputs['frame_inds'] = np.arange(0, self.audio_total_frames,
                                         2)[:, np.newaxis]
        inputs['num_clips'] = 1
        inputs['length'] = 1280
        audio_selector = AudioDecode()
        results = audio_selector(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)

    def test_load_audio_feature(self):
        target_keys = ['audios']
        inputs = copy.deepcopy(self.audio_feature_results)
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert self.check_keys_contain(results.keys(), target_keys)

        # test when no audio feature file exists
        inputs = copy.deepcopy(self.audio_feature_results)
        inputs['audio_path'] = 'foo/foo/bar.npy'
        load_audio_feature = LoadAudioFeature()
        results = load_audio_feature(inputs)
        assert results['audios'].shape == (640, 80)
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(load_audio_feature) == (
            f'{load_audio_feature.__class__.__name__}('
            f'pad_method=zero)')

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
        assert self.check_keys_contain(results.keys(), target_keys)
        assert repr(audio_feature_selector) == (
            f'{audio_feature_selector.__class__.__name__}('
            f'fix_length={128})')

    def test_pyav_decode_motion_vector(self):
        pyav_init = PyAVInit()
        pyav = PyAVDecodeMotionVector()

        # test pyav with 2-dim input
        results = {
            'filename': self.video_path,
            'frame_inds': np.arange(0, 32, 1)[:, np.newaxis]
        }
        results = pyav_init(results)
        results = pyav(results)
        target_keys = ['motion_vectors']
        assert self.check_keys_contain(results.keys(), target_keys)

        # test pyav with 1 dim input
        results = {
            'filename': self.video_path,
            'frame_inds': np.arange(0, 32, 1)
        }
        pyav_init = PyAVInit()
        results = pyav_init(results)
        pyav = PyAVDecodeMotionVector()
        results = pyav(results)

        assert self.check_keys_contain(results.keys(), target_keys)
