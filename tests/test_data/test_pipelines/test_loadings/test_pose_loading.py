import copy as cp
from collections import defaultdict

import numpy as np
from mmcv import dump
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.pipelines import (LoadKineticsPose, PoseDecode,
                                         UniformSampleFrames)


class TestPoseLoading:

    def test_uniform_sample_frames(self):
        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)

        assert str(sampling) == ('UniformSampleFrames(clip_len=8, '
                                 'num_clips=1, test_mode=True, seed=0)')
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([4, 15, 21, 24, 35, 43, 51, 63]))

        results = dict(total_frames=15, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([0, 2, 4, 6, 8, 9, 11, 13]))

        results = dict(total_frames=7, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([0, 1, 2, 3, 4, 5, 6, 0]))

        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=4, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 4
        assert_array_equal(
            sampling_results['frame_inds'],
            np.array([
                4, 15, 21, 24, 35, 43, 51, 63, 1, 11, 21, 26, 36, 47, 54, 56,
                0, 12, 18, 25, 38, 47, 55, 62, 0, 9, 21, 25, 37, 40, 49, 60
            ]))

        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=False, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert len(sampling_results['frame_inds']) == 8

    def test_pose_decode():
        kp = np.random.random([1, 16, 17, 2])
        kpscore = np.random.random([1, 16, 17])
        frame_inds = np.array([2, 4, 6, 8, 10])
        results = dict(kp=kp, kpscore=kpscore, frame_inds=frame_inds)
        pose_decode = PoseDecode()
        assert str(pose_decode) == ('PoseDecode(random_drop=False, '
                                    'random_seed=1, '
                                    'drop_prob=0.0625, '
                                    'manipulate_joints=(7, 8, 9, 10, '
                                    '13, 14, 15, 16))')
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['kp'], kp[:, frame_inds])
        assert_array_almost_equal(decode_results['kpscore'],
                                  kpscore[:, frame_inds])

        results = dict(kp=kp, kpscore=kpscore, total_frames=16)
        pose_decode = PoseDecode()
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['kp'], kp)
        assert_array_almost_equal(decode_results['kpscore'], kpscore)

        results = dict(kp=kp, kpscore=kpscore, frame_inds=frame_inds)
        pose_decode = PoseDecode(
            random_drop=True, drop_prob=1, manipulate_joints=(7, ))
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['kpscore'][..., 7], 0)

    def test_load_kinetics_pose():

        def get_mode(arr):
            cnt = defaultdict(lambda: 0)
            for num in arr:
                cnt[num] += 1
            max_val = max(cnt.values())
            return [k for k in cnt if cnt[k] == max_val], max_val

        filename = '/tmp/tmp.pkl'
        total_frames = 100
        img_shape = (224, 224)
        frame_inds = np.random.choice(range(100), size=120)
        frame_inds.sort()
        anno_flag = np.random.random(120) > 0.1
        anno_inds = np.array([i for i, f in enumerate(anno_flag) if f])
        kp = np.random.random([120, 17, 3])
        dump(kp, filename)
        results = dict(
            filename=filename,
            total_frames=total_frames,
            img_shape=img_shape,
            frame_inds=frame_inds)

        inp = cp.deepcopy(results)
        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=100, source='openpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['kp'].shape[:-1] == \
            return_results['kpscore'].shape

        num_person = return_results['kp'].shape[0]
        num_frame = return_results['kp'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert np.max(return_results['kp']) > 1
        assert num_frame == len(set(frame_inds))

        inp = cp.deepcopy(results)
        load_kinetics_pose = LoadKineticsPose(
            squeeze=False, max_person=100, source='openpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['kp'].shape[:-1] == \
            return_results['kpscore'].shape

        num_person = return_results['kp'].shape[0]
        num_frame = return_results['kp'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert np.max(return_results['kp']) > 1
        assert num_frame == total_frames

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=100, source='mmpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['kp'].shape[:-1] == \
            return_results['kpscore'].shape

        num_person = return_results['kp'].shape[0]
        num_frame = return_results['kp'].shape[1]
        assert num_person == get_mode(frame_inds[anno_inds])[1]
        assert np.max(return_results['kp']) <= 1
        assert num_frame == len(set(frame_inds[anno_inds]))

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=2, source='mmpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['kp'].shape[:-1] == \
            return_results['kpscore'].shape

        num_person = return_results['kp'].shape[0]
        num_frame = return_results['kp'].shape[1]
        assert num_person <= 2
        assert np.max(return_results['kp']) <= 1
        assert num_frame == len(set(frame_inds[anno_inds]))
