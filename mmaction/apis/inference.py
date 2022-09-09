# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Optional, Tuple, Union

import mmengine
import torch
import torch.nn as nn
import numpy as np
from mmengine.dataset import Compose, pseudo_collate
from mmengine.runner import load_checkpoint
from mmengine.utils import track_iter_progress

from mmaction.registry import MODELS
from mmaction.structures import ActionDataSample


def init_recognizer(config: Union[str, Path, mmengine.Config],
                    checkpoint: Optional[str] = None,
                    device: Union[str, torch.device] = 'cuda:0') -> nn.Module:
    """Initialize a recognizer from config file.

    Args:
        config (Union[str, :obj:`Path`, :obj:`mmengine.Config`]): Config file
            path, :obj:`Path` or the config object.
        checkpoint (str, optional): Checkpoint path/url. If set to None,
            the model will not load any weights. Defaults to None.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

    Returns:
        nn.Module: The constructed recognizer.
    """
    if isinstance(config, str):
        config = mmengine.Config.fromfile(config)
    elif not isinstance(config, mmengine.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    config.model.backbone.pretrained = None
    model = MODELS.build(config.model)

    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config
    model.to(device)
    model.eval()
    return model


def inference_recognizer(model: nn.Module,
                         video: str) -> ActionDataSample:
    """Inference a video with the recognizer.

    Args:
        model (nn.Module): The loaded recognizer.
        video (str): The video file path.

    Returns:
        :obj:`ActionDataSample`: The inference results. Specifically, the
        predicted scores are saved at ``result.pred_scores.item``.
    """
    cfg = model.cfg

    # Build the data pipeline
    val_pipeline_cfg = cfg.val_dataloader.dataset.pipeline
    if 'Init' not in val_pipeline_cfg[0]['type']:
        val_pipeline_cfg = [dict(type='OpenCVInit')] + val_pipeline_cfg
    else:
        val_pipeline_cfg[0] = dict(type='OpenCVInit')
    for i in range(len(val_pipeline_cfg)):
        if 'Decode' in val_pipeline_cfg[i]['type']:
            val_pipeline_cfg[i] = dict(type='OpenCVDecode')
    val_pipeline = Compose(val_pipeline_cfg)

    # Prepare & process inputs
    data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    data = val_pipeline(data)
    data = pseudo_collate([data])

    # Forward the model
    with torch.no_grad():
        result = model.val_step(data)[0]

    return result


def detection_inference(det_config: Union[str, Path, mmengine.Config],
                        det_checkpoint: str,
                        frame_paths: List[str],
                        det_score_thr: float = 0.9,
                        device: Union[str, torch.device] = 'cuda:0') -> list:
    """Detect human boxes given frame paths.

    Args:
        det_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`]): Config
            file path, :obj:`Path` or the config object.
        det_checkpoint: Checkpoint path/url.
        frame_paths (List[str]): The paths of frames to do detection inference.
        det_score_thr (float): The threshold of human detection score.
            Defaults to 0.9.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

    Returns:
        List[np.ndarray]: List of detected human boxes.
    """

    try:
        from mmdet.apis import inference_detector, init_detector
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_detector` and '
                          '`init_detector` from `mmdet.apis`. These apis are '
                          'required in this inference api! ')

    model = init_detector(det_config, det_checkpoint, device)
    assert model.dataset_meta['CLASSES'][0] == 'person', \
        'We require you to use a detector trained on COCO.'

    results = []
    print('Performing Human Detection for each frame')
    for frame_path in track_iter_progress(frame_paths):
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        pred_instances = result.pred_instances
        pred_instances = pred_instances[pred_instances.scores > det_score_thr]
        results.append(pred_instances.bboxes.cpu().numpy())

    return results


def pose_inference(pose_config: Union[str, Path, mmengine.Config],
                   pose_checkpoint: str,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   device: Union[str, torch.device] = 'cuda:0'):
    """Perform Top-Down pose estimation.

    Args:
        pose_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`]): Config
            file path, :obj:`Path` or the config object.
        pose_checkpoint: Checkpoint path/url.
        frame_paths (List[str]): The paths of frames to do pose inference.
        det_results (List[np.ndarray]): List of detected human boxes.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

    Returns:


    """
    try:
        from mmpose.apis import inference_topdown, init_model
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import `inference_topdown` and '
                          '`init_model` from `mmpose.apis`. These apis '
                          'are required in this inference api! ')

    model = init_model(pose_config, pose_checkpoint, device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose = inference_topdown(model, f, d, bbox_format='xyxy')[0]
        ret.append(pose)
    return ret
