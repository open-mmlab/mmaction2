# Copyright (c) OpenMMLab. All rights reserved.
import copy
import copy as cp
import os.path as osp
from collections import defaultdict

import numpy as np
import pytest
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.transforms import (DecompressPose, GeneratePoseTarget,
                                          GenSkeFeat, JointToBone,
                                          MergeSkeFeat, MMCompact, MMDecode,
                                          MMUniformSampleFrames, PadTo,
                                          PoseCompact, PoseDecode,
                                          PreNormalize2D, PreNormalize3D,
                                          ToMotion, UniformSampleFrames)


class TestPoseTransforms:

    @staticmethod
    def test_decompress_pose():

        def get_mode(arr):
            cnt = defaultdict(lambda: 0)
            for num in arr:
                cnt[num] += 1
            max_val = max(cnt.values())
            return [k for k in cnt if cnt[k] == max_val], max_val

        total_frames = 100
        img_shape = (224, 224)
        frame_inds = np.random.choice(range(100), size=120)
        frame_inds.sort()
        anno_flag = np.random.random(120) > 0.1
        anno_inds = np.array([i for i, f in enumerate(anno_flag) if f])
        kp = np.random.random([120, 17, 3])
        results = dict(
            frame_inds=frame_inds,
            keypoint=kp,
            total_frames=total_frames,
            img_shape=img_shape)

        inp = cp.deepcopy(results)

        decompress_pose = DecompressPose(squeeze=True, max_person=100)

        assert str(decompress_pose) == (
            'DecompressPose(squeeze=True, max_person=100)')
        return_results = decompress_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
               return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert num_frame == len(set(frame_inds))

        inp = cp.deepcopy(results)
        decompress_pose = DecompressPose(squeeze=False, max_person=100)
        return_results = decompress_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
               return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds)[1]
        assert num_frame == total_frames

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        decompress_pose = DecompressPose(squeeze=True, max_person=100)
        return_results = decompress_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
               return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person == get_mode(frame_inds[anno_inds])[1]
        assert num_frame == len(set(frame_inds[anno_inds]))

        inp = cp.deepcopy(results)
        inp['anno_inds'] = anno_inds
        decompress_pose = DecompressPose(squeeze=True, max_person=2)
        return_results = decompress_pose(inp)
        assert return_results['keypoint'].shape[:-1] == \
               return_results['keypoint_score'].shape

        num_person = return_results['keypoint'].shape[0]
        num_frame = return_results['keypoint'].shape[1]
        assert num_person <= 2
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
            sigma=1,
            with_kp=True,
            left_kp=(1, ),
            right_kp=(2, ),
            left_limb=(0, ),
            right_limb=(1, ),
            skeletons=())
        assert str(generate_pose_target) == ('GeneratePoseTarget(sigma=1, '
                                             'use_score=True, with_kp=True, '
                                             'with_limb=False, skeletons=(), '
                                             'double=False, left_kp=(1,), '
                                             'right_kp=(2,), left_limb=(0,), '
                                             'right_limb=(1,), scaling=1.0)')
        return_results = generate_pose_target(copy.deepcopy(results))
        assert return_results['imgs'].shape == (8, 3, 64, 64)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        results = dict(img_shape=img_shape, keypoint=kp, modality='Pose')

        generate_pose_target = GeneratePoseTarget(sigma=1, with_kp=True)
        return_results = generate_pose_target(copy.deepcopy(results))
        assert return_results['imgs'].shape == (8, 3, 64, 64)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(copy.deepcopy(results))
        assert return_results['imgs'].shape == (8, 3, 64, 64)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            double=True,
            left_limb=(0, ),
            right_limb=(1, ),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(copy.deepcopy(results))
        imgs = return_results['imgs']
        assert imgs.shape == (16, 3, 64, 64)
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
            sigma=1, with_kp=True, skeletons=())
        return_results = generate_pose_target(copy.deepcopy(results))
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
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(copy.deepcopy(results))
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
        generate_pose_target = GeneratePoseTarget(sigma=1, with_kp=True)
        return_results = generate_pose_target(copy.deepcopy(results))
        assert_array_almost_equal(return_results['imgs'], 0)

        img_shape = (64, 64)
        kp = np.array([[[[124., 124.], [140., 140.], [124., 140.]]]])
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
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)

    @staticmethod
    def test_pose_compact():
        results = {}
        results['img_shape'] = (100, 100)
        fake_kp = np.zeros([1, 4, 2, 2])
        fake_kp[:, :, 0] = [10, 10]
        fake_kp[:, :, 1] = [90, 90]
        results['keypoint'] = fake_kp

        pose_compact = PoseCompact(
            padding=0, threshold=0, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 80)
        assert str(pose_compact) == (
            'PoseCompact(padding=0, threshold=0, hw_ratio=None, '
            'allow_imgpad=False)')

        pose_compact = PoseCompact(
            padding=0.3, threshold=0, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)

        pose_compact = PoseCompact(
            padding=0.3, threshold=0, hw_ratio=None, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (104, 104)

        pose_compact = PoseCompact(
            padding=0, threshold=100, hw_ratio=None, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)

        pose_compact = PoseCompact(
            padding=0, threshold=0, hw_ratio=0.75, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 106)

    @staticmethod
    def test_pre_normalize3d():
        target_keys = ['keypoint', 'total_frames', 'body_center']

        results = dict(keypoint=np.random.randn(2, 40, 25, 3), total_frames=40)

        pre_normalize3d = PreNormalize3D(
            align_center=True, align_spine=True, align_shoulder=False)

        inp = copy.deepcopy(results)
        ret1 = pre_normalize3d(inp)

        inp = copy.deepcopy(ret1)
        ret2 = pre_normalize3d(inp)

        assert_array_equal(ret2['body_center'], np.zeros(3))
        assert_array_equal(ret1['keypoint'], ret2['keypoint'])

        pre_normalize3d = PreNormalize3D(
            align_center=True, align_spine=False, align_shoulder=True)

        inp = copy.deepcopy(results)
        ret3 = pre_normalize3d(inp)

        inp = copy.deepcopy(ret3)
        ret4 = pre_normalize3d(inp)

        assert_array_equal(ret4['body_center'], np.zeros(3))
        assert_array_equal(ret3['keypoint'], ret4['keypoint'])

        assert assert_dict_has_keys(ret1, target_keys)
        assert repr(pre_normalize3d) == 'PreNormalize3D(zaxis=[0, 1], ' \
                                        'xaxis=[8, 4], align_center=True, ' \
                                        'align_spine=False, ' \
                                        'align_shoulder=True)'

    @staticmethod
    def test_pre_normalize2d():

        def check_pose_normalize(origin_kps, target_kps, h, w):
            target_kps[..., 0] = target_kps[..., 0] * w / 2 + w / 2
            target_kps[..., 1] = target_kps[..., 1] * h / 2 + h / 2
            assert_array_almost_equal(origin_kps, target_kps, decimal=4)

        results = dict(
            keypoint=np.random.randn(1, 40, 17, 2), img_shape=(480, 854))
        pre_normalize_2d = PreNormalize2D(img_shape=(1080, 1920))
        inp = copy.deepcopy(results)
        ret1 = pre_normalize_2d(inp)
        check_pose_normalize(
            results['keypoint'], ret1['keypoint'], h=480, w=854)

        results = dict(keypoint=np.random.randn(1, 40, 17, 2))
        pre_normalize_2d = PreNormalize2D(img_shape=(1080, 1920))
        inp = copy.deepcopy(results)
        ret2 = pre_normalize_2d(inp)
        check_pose_normalize(
            results['keypoint'], ret2['keypoint'], h=1080, w=1920)

        assert repr(pre_normalize_2d) == \
               'PreNormalize2D(img_shape=(1080, 1920))'

    @staticmethod
    def test_joint_to_bone():
        with pytest.raises(ValueError):
            JointToBone(dataset='invalid')

        with pytest.raises(AssertionError):
            JointToBone()(dict(keypoint=np.random.randn(2, 15, 25, 4)))

        results = dict(keypoint=np.random.randn(2, 15, 25, 3))
        joint_to_bone = JointToBone(dataset='nturgb+d')
        center_index = 20
        results = joint_to_bone(results)
        assert_array_equal(results['keypoint'][..., center_index, :],
                           np.zeros((2, 15, 3)))

        results = dict(keypoint=np.random.randn(2, 15, 18, 3))
        joint_to_bone = JointToBone(dataset='openpose')
        center_index = 0
        center_score = results['keypoint'][..., center_index, 2]
        results = joint_to_bone(results)
        assert_array_equal(results['keypoint'][..., center_index, :2],
                           np.zeros((2, 15, 2)))
        assert_array_almost_equal(results['keypoint'][..., center_index, 2],
                                  center_score)

        results = dict(keypoint=np.random.randn(2, 15, 17, 3))
        joint_to_bone = JointToBone(dataset='coco')
        center_index = 0
        center_score = results['keypoint'][..., center_index, 2]
        results = joint_to_bone(results)
        assert_array_equal(results['keypoint'][..., center_index, :2],
                           np.zeros((2, 15, 2)))
        assert_array_almost_equal(results['keypoint'][..., center_index, 2],
                                  center_score)

        results = dict(keypoint=np.random.randn(2, 15, 17, 3))
        joint_to_bone = JointToBone(dataset='coco', target='bone')
        results = joint_to_bone(results)
        assert assert_dict_has_keys(results, ['keypoint', 'bone'])
        assert repr(joint_to_bone) == 'JointToBone(dataset=coco, target=bone)'

    @staticmethod
    def test_to_motion():
        with pytest.raises(AssertionError):
            ToMotion()(dict(keypoint=np.random.randn(2, 15, 25, 4)))

        with pytest.raises(KeyError):
            ToMotion(source='j')(dict(keypoint=np.random.randn(2, 15, 25, 4)))

        results = dict(keypoint=np.random.randn(2, 15, 25, 3))
        to_motion = ToMotion()
        results = to_motion(results)
        assert_array_equal(results['motion'][:, -1, :, :], np.zeros(
            (2, 25, 3)))
        assert assert_dict_has_keys(results, ['keypoint', 'motion'])
        assert repr(to_motion) == 'ToMotion(dataset=nturgb+d, ' \
                                  'source=keypoint, target=motion)'

    @staticmethod
    def test_merge_ske_feat():
        with pytest.raises(KeyError):
            MergeSkeFeat()(dict(b=np.random.randn(2, 15, 25, 3)))

        results = dict(
            j=np.random.randn(2, 10, 25, 3), b=np.random.randn(2, 10, 25, 3))
        merge_ske_feat = MergeSkeFeat(feat_list=['j', 'b'])
        results = merge_ske_feat(results)

        assert assert_dict_has_keys(results, ['keypoint'])
        assert results['keypoint'].shape == (2, 10, 25, 6)
        assert repr(merge_ske_feat) == "MergeSkeFeat(feat_list=['j', 'b'], " \
                                       'target=keypoint, axis=-1)'

    @staticmethod
    def test_gen_ske_feat():
        results = dict(keypoint=np.random.randn(1, 10, 25, 3))

        gen_ske_feat = GenSkeFeat(dataset='nturgb+d', feats=['j'])
        inp = copy.deepcopy(results)
        ret1 = gen_ske_feat(inp)
        assert_array_equal(ret1['keypoint'], results['keypoint'])

        gen_ske_feat = GenSkeFeat(
            dataset='nturgb+d', feats=['j', 'b', 'jm', 'bm'])
        inp = copy.deepcopy(results)
        ret2 = gen_ske_feat(inp)
        assert ret2['keypoint'].shape == (1, 10, 25, 12)

        results = dict(
            keypoint=np.random.randn(1, 10, 17, 2),
            keypoint_score=np.random.randn(1, 10, 17))
        gen_ske_feat = GenSkeFeat(dataset='coco', feats=['j', 'b', 'jm', 'bm'])
        results = gen_ske_feat(results)
        assert results['keypoint'].shape == (1, 10, 17, 12)
        assert assert_dict_has_keys(results, ['keypoint'])
        assert not assert_dict_has_keys(results, ['j', 'b', 'jm', 'bm'])
        assert repr(gen_ske_feat) == 'GenSkeFeat(dataset=coco, ' \
                                     "feats=['j', 'b', 'jm', 'bm'], axis=-1)"

    @staticmethod
    def test_uniform_sample_frames():
        results = dict(total_frames=64, start_index=0)
        sampling = UniformSampleFrames(
            clip_len=8, num_clips=1, test_mode=True, seed=0)

        assert repr(sampling) == ('UniformSampleFrames(clip_len=8, '
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
    def test_pad_to():
        with pytest.raises(AssertionError):
            PadTo(length=4, mode='invalid')

        results = dict(
            keypoint=np.random.randn(2, 3, 17, 3),
            total_frames=3,
            start_index=0)

        inp = copy.deepcopy(results)
        pad_to = PadTo(length=6, mode='loop')
        ret1 = pad_to(inp)
        kp = ret1['keypoint']
        assert_array_equal(kp[:, :3], kp[:, 3:])

        inp = copy.deepcopy(results)
        pad_to = PadTo(length=6, mode='zero')
        ret2 = pad_to(inp)
        kp = ret2['keypoint']
        assert ret2['total_frames'] == 6
        assert_array_equal(kp[:, 3:], np.zeros((2, 3, 17, 3)))

    @staticmethod
    def test_pose_decode():
        kp = np.random.random([1, 16, 17, 2])
        kpscore = np.random.random([1, 16, 17])
        frame_inds = np.array([2, 4, 6, 8, 10])
        results = dict(
            keypoint=kp, keypoint_score=kpscore, frame_inds=frame_inds)
        pose_decode = PoseDecode()
        assert repr(pose_decode) == 'PoseDecode()'
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
    def test_mm_uniform_sample_frames():
        results = dict(total_frames=64, modality='Pose')
        sampling = MMUniformSampleFrames(
            clip_len=dict(RGB=8, Pose=32), num_clips=1, test_mode=True, seed=0)
        assert repr(sampling) == ('MMUniformSampleFrames('
                                  "clip_len={'RGB': 8, 'Pose': 32}, "
                                  'num_clips=1, test_mode=True, seed=0)')

        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == dict(RGB=8, Pose=32)
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert sampling_results['modality'] == ['RGB', 'Pose']
        assert_array_equal(sampling_results['RGB_inds'],
                           np.array([4, 15, 21, 24, 35, 43, 51, 63]))
        assert_array_equal(
            sampling_results['Pose_inds'],
            np.array([
                0, 3, 5, 6, 9, 11, 13, 15, 17, 19, 21, 22, 24, 27, 28, 30, 32,
                34, 36, 39, 40, 43, 45, 46, 48, 51, 53, 55, 57, 58, 61, 62
            ]))

        results = dict(total_frames=64, modality='Pose')
        sampling = MMUniformSampleFrames(
            clip_len=dict(RGB=8, Pose=32),
            num_clips=10,
            test_mode=True,
            seed=0)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == dict(RGB=8, Pose=32)
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 10
        assert sampling_results['modality'] == ['RGB', 'Pose']
        assert len(sampling_results['RGB_inds']) == 80
        assert len(sampling_results['Pose_inds']) == 320

        results = dict(total_frames=64, modality='Pose')
        sampling = MMUniformSampleFrames(
            clip_len=dict(RGB=8, Pose=32), num_clips=1, test_mode=False)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == dict(RGB=8, Pose=32)
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert len(sampling_results['RGB_inds']) == 8
        assert len(sampling_results['Pose_inds']) == 32

    @staticmethod
    def test_mm_decode():
        mm_decode = MMDecode()

        # Pose only test
        pose_raw_results = dict(
            modality=['Pose'],
            Pose_inds=np.array([2, 4, 6, 8, 10]),
            keypoint=np.random.random([1, 16, 17, 2]),
            img_shape=(1080, 1920))
        rgb_raw_results = dict(
            modality=['RGB'],
            RGB_inds=np.array([2, 4, 6, 8, 10]),
            frame_dir=osp.join(osp.dirname(__file__), '../../data/test'))

        # test pose w/o `keypoint_score`
        mm_decode(copy.deepcopy(pose_raw_results))

        # test pose with `keypoint_score`
        pose_raw_results['keypoint_score'] = np.random.random([1, 16, 17])
        pose_results = mm_decode(copy.deepcopy(pose_raw_results))

        # test rgb
        rgb_results = mm_decode(copy.deepcopy(rgb_raw_results))

        # test pose and rgb
        pose_rgb_raw_results = {
            **rgb_raw_results,
            **pose_raw_results, 'modality': ['RGB', 'Pose']
        }
        pose_rgb_results = mm_decode(copy.deepcopy(pose_rgb_raw_results))

        assert_array_equal(pose_rgb_results['keypoint_score'],
                           pose_results['keypoint_score'])
        scaled_keypoint = copy.deepcopy(pose_results['keypoint'])
        oh, ow = pose_results['img_shape']
        nh, nw = pose_rgb_results['img_shape']
        scaled_keypoint[..., 0] *= (nw / ow)
        scaled_keypoint[..., 1] *= (nh / oh)
        assert_array_equal(pose_rgb_results['keypoint'], scaled_keypoint)
        assert_array_equal(pose_rgb_results['imgs'], rgb_results['imgs'])
        assert assert_dict_has_keys(
            pose_rgb_results, ['filename', 'img_shape', 'original_shape'])
        assert repr(mm_decode) == 'MMDecode(io_backend=disk)'

    @staticmethod
    def test_mm_compact():
        results = {}
        results['img_shape'] = (100, 100)
        fake_kp = np.zeros([1, 4, 2, 2])
        fake_kp[:, :, 0] = [10, 10]
        fake_kp[:, :, 1] = [90, 90]
        results['keypoint'] = fake_kp
        results['imgs'] = list(np.zeros([3, 100, 100, 3]))

        pose_compact = MMCompact(
            padding=0, threshold=0, hw_ratio=1, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 80)
        assert ret['imgs'][0].shape[:-1] == (80, 80)
        assert str(pose_compact) == (
            'MMCompact(padding=0, threshold=0, hw_ratio=(1, 1), '
            'allow_imgpad=False)')

        pose_compact = MMCompact(
            padding=0.3, threshold=0, hw_ratio=1, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)
        assert ret['imgs'][0].shape[:-1] == (100, 100)

        pose_compact = MMCompact(
            padding=0.3, threshold=0, hw_ratio=1, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (104, 104)
        assert ret['imgs'][0].shape[:-1] == (104, 104)

        pose_compact = MMCompact(
            padding=0, threshold=100, hw_ratio=1, allow_imgpad=False)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (100, 100)
        assert ret['imgs'][0].shape[:-1] == (100, 100)

        pose_compact = MMCompact(
            padding=0, threshold=0, hw_ratio=0.75, allow_imgpad=True)
        inp = copy.deepcopy(results)
        ret = pose_compact(inp)
        assert ret['img_shape'] == (80, 106)
        assert ret['imgs'][0].shape[:-1] == (80, 106)
