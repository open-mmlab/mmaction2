# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def init_recognizer(config: Union[str, Path, mmengine.Config],
                    checkpoint: Optional[str] = None,
                    device: Union[str, torch.device] = 'cuda:0') -> nn.Module:
    """Initialize a recognizer from config file.

    Args:
        config (str or :obj:`Path` or :obj:`mmengine.Config`): Config file
            path, :obj:`Path` or the config object.
        checkpoint (str, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Defaults to None.
        device (str | torch.device): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, (str, Path)):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    init_default_scope(config.get('default_scope', 'mmaction'))

    if hasattr(config.model, 'backbone') and config.model.backbone.get(
            'pretrained', None):
        config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model: nn.Module,
                         video: Union[str, dict],
                         test_pipeline: Optional[Compose] = None
                         ) -> ActionDataSample:
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (Union[str, dict]): The video file path or the results
            dictionary (the input of pipeline).
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_score``.
    """

    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, str) and osp.exists(video):
        if video.endswith('.npy'):
            input_flag = 'audio'
        else:
            input_flag = 'video'
    else:
        raise RuntimeError(f'The type of argument `video` is not supported: '
                           f'{type(video)}')

    if input_flag == 'dict':
        data = video
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    if input_flag == 'audio':
        data = dict(
            audio_path=video,
            total_frames=len(np.load(video)),
            start_index=0,
            label=-1)

    data = test_pipeline(data)
    data = pseudo_collate([data])

    # Forward the model
    with torch.no_grad():
        result = model.test_step(data)[0]

    return result


def inference_skeleton(model: nn.Module,
                       pose_results: List[dict],
                       img_shape: Tuple[int],
                       test_pipeline: Optional[Compose] = None
                       ) -> ActionDataSample:
    """Inference a pose results with the skeleton recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        pose_results (List[dict]): The pose estimation results dictionary
            (the results of `pose_inference`)
        img_shape (Tuple[int]): The original image shape used for inference
            skeleton recognizer.
        test_pipeline (:obj:`Compose`, optional): The test pipeline.
            If not specified, the test pipeline in the config will be
            used. Defaults to None.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_score``.
    """
    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    h, w = img_shape
    num_keypoint = pose_results[0]['keypoints'].shape[1]
    num_frame = len(pose_results)
    num_person = max([len(x['keypoints']) for x in pose_results])
    fake_anno = dict(
        frame_dict='',
        label=-1,
        img_shape=(h, w),
        origin_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    keypoint = np.zeros((num_frame, num_person, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_frame, num_person, num_keypoint),
                              dtype=np.float16)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoint[f_idx, p_idx] = frm_pose['keypoints'][p_idx]
            keypoint_score[f_idx, p_idx] = frm_pose['keypoint_scores'][p_idx]

    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))
    return inference_recognizer(model, fake_anno, test_pipeline)


def detection_inference(det_config: Union[str, Path, mmengine.Config,
                                          nn.Module],
                        det_checkpoint: str,
                        frame_paths: List[str],
                        det_score_thr: float = 0.9,
                        det_cat_id: int = 0,
                        device: Union[str, torch.device] = 'cuda:0',
                        with_score: bool = False) -> tuple:
    """Detect human boxes given frame paths.

    Args:
        det_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`,
            :obj:`torch.nn.Module`]):
            Det config file path or Detection model object. It can be
            a :obj:`Path`, a config object, or a module object.
        det_checkpoint: Checkpoint path/url.
        frame_paths (List[str]): The paths of frames to do detection inference.
        det_score_thr (float): The threshold of human detection score.
            Defaults to 0.9.
        det_cat_id (int): The category id for human detection. Defaults to 0.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.
        with_score (bool): Whether to append detection score after box.
            Defaults to None.

    Returns:
        List[np.ndarray]: List of detected human boxes.
        List[:obj:`DetDataSample`]: List of data samples, generally used
            to visualize data.
    """
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmdet.structures import DetDataSample
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_detector` and '
                          '`init_detector` from `mmdet.apis`. These apis are '
                          'required in this inference api! ')
    if isinstance(det_config, nn.Module):
        model = det_config
    else:
        model = init_detector(
            config=det_config, checkpoint=det_checkpoint, device=device)

    results = []
    data_samples = []
    print('Performing Human Detection for each frame')
    for frame_path in track_iter_progress(frame_paths):
        det_data_sample: DetDataSample = inference_detector(model, frame_path)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        scores = pred_instance.scores
        # We only keep human detection bboxs with score larger
        # than `det_score_thr` and category id equal to `det_cat_id`.
        valid_idx = np.logical_and(pred_instance.labels == det_cat_id,
                                   pred_instance.scores > det_score_thr)
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]

        if with_score:
            bboxes = np.concatenate((bboxes, scores[:, None]), axis=-1)
        results.append(bboxes)
        data_samples.append(det_data_sample)

    return results, data_samples


def pose_inference(pose_config: Union[str, Path, mmengine.Config, nn.Module],
                   pose_checkpoint: str,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   device: Union[str, torch.device] = 'cuda:0') -> tuple:
    """Perform Top-Down pose estimation.

    Args:
        pose_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`,
            :obj:`torch.nn.Module`]): Pose config file path or
            pose model object. It can be a :obj:`Path`, a config object,
            or a module object.
        pose_checkpoint: Checkpoint path/url.
        frame_paths (List[str]): The paths of frames to do pose inference.
        det_results (List[np.ndarray]): List of detected human boxes.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

    Returns:
        List[List[Dict[str, np.ndarray]]]: List of pose estimation results.
        List[:obj:`PoseDataSample`]: List of data samples, generally used
            to visualize data.
    """
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample, merge_data_samples
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')
    if isinstance(pose_config, nn.Module):
        model = pose_config
    else:
        model = init_model(pose_config, pose_checkpoint, device)

    results = []
    data_samples = []
    print('Performing Human Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample] \
            = inference_topdown(model, f, d[..., :4], bbox_format='xyxy')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = model.dataset_meta
        # make fake pred_instances
        if not hasattr(pose_data_sample, 'pred_instances'):
            num_keypoints = model.dataset_meta['num_keypoints']
            pred_instances_data = dict(
                keypoints=np.empty(shape=(0, num_keypoints, 2)),
                keypoints_scores=np.empty(shape=(0, 17), dtype=np.float32),
                bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                bbox_scores=np.empty(shape=(0), dtype=np.float32))
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)

        poses = pose_data_sample.pred_instances.to_dict()
        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples
