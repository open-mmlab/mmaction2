# Copyright (c) OpenMMLab. All rights reserved.
import copy
import copy as cp
import os.path as osp
import tempfile
from collections import defaultdict

import numpy as np
import pytest
from mmengine import dump
from mmengine.testing import assert_dict_has_keys
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mmaction.datasets.transforms import (GeneratePoseTarget, LoadKineticsPose,
                                          PaddingWithLoop, PoseDecode,
                                          JointToBone,
                                          ToMotion, MergeSkeFeat, GenSkeFeat)


class TestPoseLoading:

    @staticmethod
    def test_load_kinetics_pose():
        def get_mode(arr):
            cnt = defaultdict(lambda: 0)
            for num in arr:
                cnt[num] += 1
            max_val = max(cnt.values())
            return [k for k in cnt if cnt[k] == max_val], max_val

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = osp.join(tmpdir, 'tmp.pkl')
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
                squeeze=True, max_person=100, source='openpose-18')

            assert str(load_kinetics_pose) == (
                'LoadKineticsPose(io_backend=disk, '
                'squeeze=True, max_person=100, '
                "keypoint_weight={'face': 1, "
                "'torso': 2, 'limb': 3}, "
                'source=openpose-18, kwargs={})')
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
                squeeze=False, max_person=100, source='openpose-18')
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
            sigma=1, with_kp=True, left_kp=(0,), right_kp=(1,), skeletons=())
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
            sigma=1, with_kp=True, left_kp=(0,), right_kp=(1,), skeletons=())
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 3)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=False,
            with_limb=True,
            left_kp=(0,),
            right_kp=(1,),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert return_results['imgs'].shape == (8, 64, 64, 3)
        assert_array_almost_equal(return_results['imgs'][0],
                                  return_results['imgs'][1])

        generate_pose_target = GeneratePoseTarget(
            sigma=1,
            with_kp=True,
            with_limb=True,
            left_kp=(0,),
            right_kp=(1,),
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
            left_kp=(0,),
            right_kp=(1,),
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
            sigma=1, with_kp=True, left_kp=(0,), right_kp=(1,), skeletons=())
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
            left_kp=(0,),
            right_kp=(1,),
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
            sigma=1, with_kp=True, left_kp=(0,), right_kp=(1,), skeletons=())
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
            left_kp=(0,),
            right_kp=(1,),
            skeletons=((0, 1), (1, 2), (0, 2)))
        return_results = generate_pose_target(results)
        assert_array_almost_equal(return_results['imgs'], 0)

    @staticmethod
    def test_padding_with_loop():
        results = dict(total_frames=3, start_index=0)
        sampling = PaddingWithLoop(clip_len=6)
        sampling_results = sampling(results)
        assert sampling_results['clip_len'] == 6
        assert sampling_results['frame_interval'] is None
        assert sampling_results['num_clips'] == 1
        assert_array_equal(sampling_results['frame_inds'],
                           np.array([0, 1, 2, 0, 1, 2]))

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
        assert_array_equal(results['motion'][:, -1, :, :], np.zeros((2, 25, 3)))
        assert assert_dict_has_keys(results, ['keypoint', 'motion'])
        assert repr(to_motion) == 'ToMotion(dataset=nturgb+d, ' \
                                  'source=keypoint, target=motion)'

    @staticmethod
    def test_merge_ske_feat():
        with pytest.raises(KeyError):
            MergeSkeFeat()(dict(b=np.random.randn(2, 15, 25, 3)))

        results = dict(j=np.random.randn(2, 10, 25, 3),
                       b=np.random.randn(2, 10, 25, 3))
        merge_ske_feat = MergeSkeFeat(feat_list=['j', 'b'])
        results = merge_ske_feat(results)

        assert assert_dict_has_keys(results, ['keypoint'])
        assert results['keypoint'].shape == (2, 10, 25, 6)
        assert repr(merge_ske_feat) == "MergeSkeFeat(feat_list=['j', 'b'], " \
                                       "target=keypoint, axis=-1)"

    @staticmethod
    def test_gen_ske_feat():
        results = dict(keypoint=np.random.randn(1, 10, 25, 3))

        gen_ske_feat = GenSkeFeat(dataset='nturgb+d', feats=['j'])
        inp = copy.deepcopy(results)
        ret1 = gen_ske_feat(inp)
        assert_array_equal(ret1['keypoint'], results['keypoint'])

        gen_ske_feat = GenSkeFeat(dataset='nturgb+d',
                                  feats=['j', 'b', 'jm', 'bm'])
        inp = copy.deepcopy(results)
        ret2 = gen_ske_feat(inp)
        assert ret2['keypoint'].shape == (1, 10, 25, 12)

        results = dict(keypoint=np.random.randn(1, 10, 17, 2),
                       keypoint_score=np.random.randn(1, 10, 17))
        gen_ske_feat = GenSkeFeat(dataset='coco',
                                  feats=['j', 'b', 'jm', 'bm'])
        results = gen_ske_feat(results)
        assert results['keypoint'].shape == (1, 10, 17, 12)
        assert assert_dict_has_keys(results, ['keypoint'])
        assert not assert_dict_has_keys(results, ['j', 'b', 'jm', 'bm'])
        assert repr(gen_ske_feat) == "GenSkeFeat(dataset=coco, " \
                                     "feats=['j', 'b', 'jm', 'bm'], axis=-1)"

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
        assert_array_almost_equal(decode_results['keypoint'],
                                  kp[:, frame_inds])
        assert_array_almost_equal(decode_results['keypoint_score'],
                                  kpscore[:, frame_inds])

        results = dict(keypoint=kp, keypoint_score=kpscore, total_frames=16)
        pose_decode = PoseDecode()
        decode_results = pose_decode(results)
        assert_array_almost_equal(decode_results['keypoint'], kp)
        assert_array_almost_equal(decode_results['keypoint_score'], kpscore)


def check_pose_normalize(origin_keypoints, result_keypoints, norm_cfg):
    target_keypoints = result_keypoints.copy()
    target_keypoints *= (norm_cfg['max_value'] - norm_cfg['min_value'])
    target_keypoints += norm_cfg['mean']
    assert_array_almost_equal(origin_keypoints, target_keypoints, decimal=4)
