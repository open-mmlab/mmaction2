# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from mmengine.testing import assert_dict_has_keys
from parameterized import parameterized

from mmaction.apis import (detection_inference, inference_recognizer,
                           init_recognizer, pose_inference)
from mmaction.structures import ActionDataSample
from mmaction.utils import frame_extract, get_str_type


class TestInference(TestCase):

    @parameterized.expand([(('configs/recognition/tsn/'
                             'tsn_imagenet-pretrained-r50_8xb32-'
                             '1x1x3-100e_kinetics400-rgb.py'), ('cpu', 'cuda'))
                           ])
    def test_init_recognizer(self, config, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue

            # test `init_recognizer` with str path
            _ = init_recognizer(config_file, device=device)

            # test `init_recognizer` with :obj:`Path`
            _ = init_recognizer(Path(config_file), device=device)

            # test `init_recognizer` with undesirable type
            with self.assertRaisesRegex(
                    TypeError, 'config must be a filename or Config object'):
                config_list = [config_file]
                _ = init_recognizer(config_list)

    @parameterized.expand([(('configs/recognition/tsn/'
                             'tsn_imagenet-pretrained-r50_8xb32-'
                             '1x1x3-100e_kinetics400-rgb.py'), 'demo/demo.mp4',
                            ('cpu', 'cuda'))])
    def test_inference_recognizer(self, config, video_path, devices):
        project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
        project_dir = osp.join(project_dir, '..')
        config_file = osp.join(project_dir, config)
        video_path = osp.join(project_dir, video_path)

        for device in devices:
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            model = init_recognizer(config_file, device=device)

            for ops in model.cfg.test_pipeline:
                if get_str_type(ops['type']) in ('TenCrop', 'ThreeCrop'):
                    # Use CenterCrop to reduce memory in order to pass CI
                    ops['type'] = 'CenterCrop'

            result = inference_recognizer(model, video_path)

            self.assertIsInstance(result, ActionDataSample)
            self.assertTrue(result.pred_score.shape, (400, ))

    def test_detection_inference(self):
        from mmdet.apis import init_detector
        from mmdet.structures import DetDataSample

        for device in ('cpu', 'cuda'):
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
            project_dir = osp.join(project_dir, '..')
            det_config = 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'  # noqa: E501
            det_ckpt = 'http://download.openmmlab.com/mmdetection/' \
                       'v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'  # noqa: E501
            video_path = 'demo/demo_skeleton.mp4'
            video_path = osp.join(project_dir, video_path)
            config_file = osp.join(project_dir, det_config)
            with TemporaryDirectory() as tmpdir:
                frm_paths, _ = frame_extract(video_path, out_dir=tmpdir)
                # skip remaining frames to speed up ut
                frm_paths = frm_paths[:10]
                results, data_samples = detection_inference(
                    config_file, det_ckpt, frm_paths, device=device)
                self.assertTrue(results[0].shape, (4, ))
                self.assertIsInstance(data_samples[0], DetDataSample)
                # test with_score
                results, data_samples = detection_inference(
                    config_file,
                    det_ckpt,
                    frm_paths,
                    with_score=True,
                    device=device)
                self.assertTrue(results[0].shape, (5, ))
                # test inference with model object
                model = init_detector(
                    config=det_config, checkpoint=det_ckpt, device=device)
                results, data_samples = detection_inference(
                    model, None, frm_paths, device=device)
                self.assertTrue(results[0].shape, (4, ))
                self.assertIsInstance(data_samples[0], DetDataSample)

    def test_pose_inference(self):
        from mmpose.apis import init_model
        from mmpose.structures import PoseDataSample

        for device in ('cpu', 'cuda'):
            if device == 'cuda' and not torch.cuda.is_available():
                # Skip the test if cuda is required but unavailable
                continue
            project_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
            project_dir = osp.join(project_dir, '..')
            det_config = 'demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'  # noqa: E501
            det_ckpt = 'http://download.openmmlab.com/mmdetection/' \
                       'v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'  # noqa: E501
            pose_config = 'demo/demo_configs/' \
                          'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
            pose_ckpt = 'https://download.openmmlab.com/mmpose/top_down/' \
                        'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
            video_path = 'demo/demo_skeleton.mp4'
            video_path = osp.join(project_dir, video_path)
            pose_config = osp.join(project_dir, pose_config)
            with TemporaryDirectory() as tmpdir:
                frm_paths, _ = frame_extract(video_path, out_dir=tmpdir)
                # skip remaining frames to speed up ut
                frm_paths = frm_paths[:10]
                det_results, _ = detection_inference(
                    det_config, det_ckpt, frm_paths, device=device)

                results, data_samples = pose_inference(
                    pose_config,
                    pose_ckpt,
                    frm_paths,
                    det_results,
                    device=device)
                assert_dict_has_keys(results[0], ('keypoints', 'bbox_scores',
                                                  'bboxes', 'keypoint_scores'))
                self.assertIsInstance(data_samples[0], PoseDataSample)

                # test inference with model object
                model = init_model(
                    config=pose_config, checkpoint=pose_ckpt, device=device)
                results, data_samples = pose_inference(
                    model, None, frm_paths, det_results, device=device)
                assert_dict_has_keys(results[0], ('keypoints', 'bbox_scores',
                                                  'bboxes', 'keypoint_scores'))
                self.assertIsInstance(data_samples[0], PoseDataSample)
