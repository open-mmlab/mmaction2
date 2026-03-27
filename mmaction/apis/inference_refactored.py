"""
Copyright (c) OpenMMLab. All rights reserved.
Refactored version with improved error handling and organization.
"""
import os.path as osp
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

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
        checkpoint (str, optional): Checkpoint path/url. Defaults to None.
        device (str | torch.device): The desired device. Defaults to 'cuda:0'.

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


def create_empty_pose_result(num_keypoints: int) -> Dict:
    """Create empty pose result with proper dimensions.

    Args:
        num_keypoints (int): Number of keypoints in the model.

    Returns:
        Dict: Empty pose result dictionary.
    """
    return {
        'keypoints': np.empty(shape=(0, num_keypoints, 2)),
        'keypoint_scores': np.empty(shape=(0, num_keypoints)),
        'bboxes': np.empty(shape=(0, 4)),
        'bbox_scores': np.empty(shape=(0))
    }


def validate_inputs(frame_paths: List[str], det_results: List[np.ndarray]):
    """Validate input parameters for pose inference.

    Args:
        frame_paths (List[str]): List of frame paths.
        det_results (List[np.ndarray]): List of detection results.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")
    if len(frame_paths) != len(det_results):
        raise ValueError(
            f"Number of frames ({len(frame_paths)}) must match "
            f"number of detection results ({len(det_results)})")


def process_pose_data(pose_data_samples: List['PoseDataSample'],
                      model_meta: Dict,
                      num_keypoints: int) -> Tuple['PoseDataSample', Dict]:
    """Process pose data samples and create pose results.

    Args:
        pose_data_samples (List[PoseDataSample]): List of pose data samples.
        model_meta (Dict): Model metadata.
        num_keypoints (int): Number of keypoints.

    Returns:
        Tuple[PoseDataSample, Dict]: Processed pose data sample and poses dict.
    """
    from mmpose.structures import merge_data_samples
    pose_data_sample = merge_data_samples(pose_data_samples)
    pose_data_sample.dataset_meta = model_meta

    if not hasattr(pose_data_sample, 'pred_instances'):
        pred_instances_data = create_empty_pose_result(num_keypoints)
        pose_data_sample.pred_instances = InstanceData(**pred_instances_data)

    return pose_data_sample, pose_data_sample.pred_instances.to_dict()


def pose_inference(pose_config: Union[str, Path, mmengine.Config, nn.Module],
                   pose_checkpoint: str,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   device: Union[str, torch.device] = 'cuda:0') -> tuple:
    """Perform Top-Down pose estimation with improved error handling.

    Args:
        pose_config: Pose config file path or pose model object.
        pose_checkpoint: Checkpoint path/url.
        frame_paths: The paths of frames to do pose inference.
        det_results: List of detected human boxes.
        device: The desired device. Defaults to 'cuda:0'.

    Returns:
        Tuple[List[Dict], List[PoseDataSample]]: Pose results and data samples.
    """
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample
    except ImportError:
        raise ImportError('Failed to import required mmpose components')

    validate_inputs(frame_paths, det_results)

    # Initialize model
    model = pose_config if isinstance(pose_config, nn.Module) else \
        init_model(pose_config, pose_checkpoint, device)
    num_keypoints = model.dataset_meta['num_keypoints']

    results = []
    data_samples = []
    print('Performing Human Pose Estimation for each frame')

    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        if d.size == 0:
            # Handle empty detection results
            pred_instances_data = create_empty_pose_result(num_keypoints)
            pose_data_sample = PoseDataSample()
            pose_data_sample.pred_instances = InstanceData(
                **pred_instances_data)
            pose_data_sample.dataset_meta = model.dataset_meta
            results.append(pred_instances_data)
            data_samples.append(pose_data_sample)
            continue

        if d.shape[1] < 4:
            raise ValueError(
                f"Detection boxes must have at least 4 values, got shape {d.shape}")

        pose_data_samples = inference_topdown(
            model, f, d[..., :4], bbox_format='xyxy')
        pose_data_sample, poses = process_pose_data(
            pose_data_samples, model.dataset_meta, num_keypoints)

        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples


def prepare_skeleton_data(pose_results: List[dict],
                          img_shape: Tuple[int]) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Prepare data for skeleton inference.

    Args:
        pose_results: List of pose estimation results.
        img_shape: Original image shape.

    Returns:
        Tuple containing fake annotation, keypoint array, and keypoint scores.
    """
    h, w = img_shape
    num_keypoint = pose_results[0]['keypoints'].shape[1]
    num_frame = len(pose_results)
    num_person = max([len(x['keypoints']) for x in pose_results])

    fake_anno = {
        'frame_dict': '',
        'label': -1,
        'img_shape': (h, w),
        'origin_shape': (h, w),
        'start_index': 0,
        'modality': 'Pose',
        'total_frames': num_frame
    }

    keypoint = np.zeros(
        (num_frame, num_person, num_keypoint, 2), dtype=np.float16)
    keypoint_score = np.zeros(
        (num_frame, num_person, num_keypoint), dtype=np.float16)

    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoint[f_idx, p_idx] = frm_pose['keypoints'][p_idx]
            keypoint_score[f_idx, p_idx] = frm_pose['keypoint_scores'][p_idx]

    return fake_anno, keypoint, keypoint_score


def inference_skeleton(model: nn.Module,
                       pose_results: List[dict],
                       img_shape: Tuple[int],
                       test_pipeline: Optional[Compose] = None) -> ActionDataSample:
    """Inference a pose results with the skeleton recognizer.

    Args:
        model: The loaded recognizer.
        pose_results: The pose estimation results dictionary.
        img_shape: The original image shape.
        test_pipeline: The test pipeline. Defaults to None.

    Returns:
        ActionDataSample: The inference results.
    """
    if not pose_results:
        raise ValueError("pose_results cannot be empty")

    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline = Compose(cfg.test_pipeline)

    fake_anno, keypoint, keypoint_score = prepare_skeleton_data(
        pose_results, img_shape)

    # Transpose keypoint data to match expected format
    fake_anno['keypoint'] = keypoint.transpose((1, 0, 2, 3))
    fake_anno['keypoint_score'] = keypoint_score.transpose((1, 0, 2))

    return inference_recognizer(model, fake_anno, test_pipeline)


def inference_recognizer(model: nn.Module,
                         video: Union[str, dict],
                         test_pipeline: Optional[Compose] = None) -> ActionDataSample:
    """Inference a video with the recognizer.

    Args:
        model: The loaded recognizer.
        video: Video file path or results dictionary.
        test_pipeline: The test pipeline. Defaults to None.

    Returns:
        ActionDataSample: The inference results.
    """
    if test_pipeline is None:
        cfg = model.cfg
        init_default_scope(cfg.get('default_scope', 'mmaction'))
        test_pipeline = Compose(cfg.test_pipeline)

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, str) and osp.exists(video):
        input_flag = 'audio' if video.endswith('.npy') else 'video'
    else:
        raise RuntimeError(f'Unsupported video type: {type(video)}')

    if input_flag == 'dict':
        data = video
    elif input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    else:  # audio
        data = dict(
            audio_path=video,
            total_frames=len(np.load(video)),
            start_index=0,
            label=-1)

    data = test_pipeline(data)
    data = pseudo_collate([data])

    with torch.no_grad():
        result = model.test_step(data)[0]

    return result
