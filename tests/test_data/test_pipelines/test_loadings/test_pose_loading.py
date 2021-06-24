import copy as cp
from collections import defaultdict

import numpy as np
import pytest
from mmcv import dump
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.pipelines import (GeneratePoseTarget, LoadKineticsPose,
                                         PoseDecode, UniformSampleFrames)


class TestPoseLoading:

    @staticmethod
    def test_uniform_sample_frames():
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

        results = dict(total_frames=7, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=8, test_mode=True, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 8
        assert len(sampling_results['frame_inds']) == 64

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

        results = dict(total_frames=7, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=False, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert len(sampling_results['frame_inds']) == 8

        results = dict(total_frames=15, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=False, seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 8
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert len(sampling_results['frame_inds']) == 8

    @staticmethod
    def test_pose_decode():
        kp = np.random.random([1, 16, 17, 2])
        kpscore = np.random.random([1, 16, 17])
        frame_inds = np.array([2, 4, 6, 8, 10])
        results = dict(
            keypoint=kp, keypoint_score=kpscore, frame_inds=frame_inds)
        pose_decode = PoseDecode()
        assert str(pose_decode) == ('PoseDecode()')
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['keypoint'], kp[:,
                                                                 frame_inds])
        assert_array_almost_equal(decode_results['keypoint_score'],
                                  kpscore[:, frame_inds])

        results = dict(keypoint=kp, keypoint_score=kpscore, total_frames=16)
        pose_decode = PoseDecode()
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['keypoint'], kp)
        assert_array_almost_equal(decode_results['keypoint_score'], kpscore)

    @staticmethod
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

        with pytest.raises(NotImplementedError):
            LoadKineticsPose(squeeze=True, max_person=100, source='xxx')

        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=100, source='openpose')

        assert str(load_kinetics_pose) == ('LoadKineticsPose(io_backend=disk, '
                                           'squeeze=True, max_person=100, '
                                           "keypoint_weight={'face': 1, "
                                           "'torso': 2, 'limb': 3}, "
                                           'source=openpose, kwargs={})')
        return_results = load_kinetics_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
            return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert np.max(return_results['keypoint']) > 1
        assert num_frame == len(set(frame_inds))

        inp = cp.deepcopy(results)
        load_kinetics_pose = LoadKineticsPose(
            squeeze=False, max_person=100, source='openpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
            return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert np.max(return_results['keypoint']) > 1
        assert num_frame == total_frames

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=100, source='mmpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
            return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds[anno_inds])[1]
        assert np.max(return_results['keypoint']) <= 1
        assert num_frame == len(set(frame_inds[anno_inds]))

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        load_kinetics_pose = LoadKineticsPose(
            squeeze=True, max_person=2, source='mmpose')
        return_results = load_kinetics_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
            return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person <= 2
        assert np.max(return_results['keypoint']) <= 1
        assert num_frame == len(set(frame_inds[anno_inds]))

    @staticmethod
    def test_generate_pose_target():
        img_shape = (64, 64)
        kp = np.array([[[[24, 24], [40, 40], [24, 40]]]])
        kpscore = np.array([[[1., 1., 1.]]])
        kp = np.concatenate([kp] * 8, axis=1)
        kpscore = np.concatenate([kpscore] * 8, axis=1)
        results = dict(
            img_shape=img_shape,
            keypoint=kp,
            keypoint_score=kpscore,
            modality='Pose')

        generate_pose_target = GeneratePoseTarget(
            sigma=1, with_kp=True, left_kp=(0, ), right_kp=(1, ), skeletons=())
        assert str(generate_pose_target) == ('GeneratePoseTarget(sigma=1, '
                                             'use_score=True, with_kp=True, '
                                             'with_limb=False, skeletons=(), '
                                             'double=False, left_kp=(0,), '
                                             'right_kp=(1,))')
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 3)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        results = dict(img_shape=img_shape, keypoint=kp, modality='Pose')

        generate_pose_target = GeneratePoseTarget(
            sigma=1, with_kp=True, left_kp=(0, ), right_kp=(1, ), skeletons=())
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 3)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            left_kp=(0, ),
            right_kp=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 3)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=True,
            with_limb=True,
            left_kp=(0, ),
            right_kp=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 6)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=True,
            with_limb=True,
            double=True,
            left_kp=(0, ),
            right_kp=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        imgs = return_results['imgs']
        assert imgs.shape == (16, 64, 64, 6)
        assert_array_almost_equal(imgs[0], imgs[1])
        assert_array_almost_equal(imgs[:8, 2], imgs[8:, 2, :, ::-1])
        assert_array_almost_equal(imgs[:8, 0], imgs[8:, 1, :, ::-1])
        assert_array_almost_equal(imgs[:8, 1], imgs[8:, 0, :, ::-1])

        img_shape = (64, 64)
        kp = np.array([[[[24, 24], [40, 40], [24, 40]]]])
        kpscore = np.array([[[0., 0., 0.]]])
        kp = np.concatenate([kp] * 8, axis=1)
        kpscore = np.concatenate([kpscore] * 8, axis=1)
        results = dict(
            img_shape=img_shape,
            keypoint=kp,
            keypoint_score=kpscore,
            modality='Pose')
        generate_pose_target = GeneratePoseTarget(
            sigma=1, with_kp=True, left_kp=(0, ), right_kp=(1, ), skeletons=())
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)

        img_shape = (64, 64)
        kp = np.array([[[[24, 24], [40, 40], [24, 40]]]])
        kpscore = np.array([[[0., 0., 0.]]])
        kp = np.concatenate([kp] * 8, axis=1)
        kpscore = np.concatenate([kpscore] * 8, axis=1)
        results = dict(
            img_shape=img_shape,
            keypoint=kp,
            keypoint_score=kpscore,
            modality='Pose')
        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            left_kp=(0, ),
            right_kp=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)

        img_shape = (64, 64)
        kp = np.array([[[[124, 124], [140, 140], [124, 140]]]])
        kpscore = np.array([[[0., 0., 0.]]])
        kp = np.concatenate([kp] * 8, axis=1)
        kpscore = np.concatenate([kpscore] * 8, axis=1)
        results = dict(
            img_shape=img_shape,
            keypoint=kp,
            keypoint_score=kpscore,
            modality='Pose')
        generate_pose_target = GeneratePoseTarget(
            sigma=1, with_kp=True, left_kp=(0, ), right_kp=(1, ), skeletons=())
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)

        img_shape = (64, 64)
        kp = np.array([[[[124, 124], [140, 140], [124, 140]]]])
        kpscore = np.array([[[0., 0., 0.]]])
        kp = np.concatenate([kp] * 8, axis=1)
        kpscore = np.concatenate([kpscore] * 8, axis=1)
        results = dict(
            img_shape=img_shape,
            keypoint=kp,
            keypoint_score=kpscore,
            modality='Pose')
        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            left_kp=(0, ),
            right_kp=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)
