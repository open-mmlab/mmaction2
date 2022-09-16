# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List, Optional, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
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
    if isinstance(config, (str, Path)):
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
        predicted scores are saved at ``result.pred_scores.item``.
    """

    if test_pipeline is None:
        cfg = model.cfg
        test_pipeline_cfg = cfg.test_pipeline
        test_pipeline = Compose(test_pipeline_cfg)

    input_flag = None
    if isinstance(video, dict):
        input_flag = 'dict'
    elif isinstance(video, str):
        input_flag = 'video'
    else:
        raise RuntimeError(f'The type of argument `video` is not supported: '
                           f'{type(video)}')

    if input_flag == 'dict':
        data = video
    if input_flag == 'video':
        data = dict(filename=video, label=-1, start_index=0, modality='RGB')

    data = test_pipeline(data)
    data = pseudo_collate([data])

    # Forward the model
    with torch.no_grad():
        result = model.test_step(data)[0]

    return result


def detection_inference(det_config: Union[str, Path, mmengine.Config],
                        det_checkpoint: str,
                        frame_paths: List[str],
                        det_score_thr: float = 0.9,
                        det_cat_id: int = 0,
                        device: Union[str, torch.device] = 'cuda:0') -> tuple:
    """Detect human boxes given frame paths.

    Args:
        det_config (Union[str, :obj:`Path`, :obj:`mmengine.Config`]): Config
            file path, :obj:`Path` or the config object.
        det_checkpoint: Checkpoint path/url.
        frame_paths (List[str]): The paths of frames to do detection inference.
        det_score_thr (float): The threshold of human detection score.
            Defaults to 0.9.
        det_cat_id (int): The category id for human detection. Defaults to 0.
        device (Union[str, torch.device]): The desired device of returned
            tensor. Defaults to ``'cuda:0'``.

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

    model = init_detector(det_config, det_checkpoint, device)

    results = []
    data_samples = []
    print('Performing Human Detection for each frame')
    for frame_path in track_iter_progress(frame_paths):
        det_data_sample: DetDataSample = inference_detector(model, frame_path)
        pred_instance = det_data_sample.pred_instances.cpu().numpy()
        bboxes = pred_instance.bboxes
        # We only keep human detection bboxs with score larger
        # than `det_score_thr` and category id equal to `det_cat_id`.
        bboxes = bboxes[np.logical_and(pred_instance.labels == det_cat_id,
                                       pred_instance.scores > det_score_thr)]
        results.append(bboxes)
        data_samples.append(det_data_sample)

    return results, data_samples


def pose_inference(pose_config: Union[str, Path, mmengine.Config],
                   pose_checkpoint: str,
                   frame_paths: List[str],
                   det_results: List[np.ndarray],
                   device: Union[str, torch.device] = 'cuda:0') -> tuple:
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

    model = init_model(pose_config, pose_checkpoint, device)

    results = []
    data_samples = []
    print('Performing Human Pose Estimation for each frame')
    for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
        pose_data_samples: List[PoseDataSample] \
            = inference_topdown(model, f, d, bbox_format='xyxy')
        pose_data_sample = merge_data_samples(pose_data_samples)
        pose_data_sample.dataset_meta = model.dataset_meta
        poses = pose_data_sample.pred_instances.to_dict()
        results.append(poses)
        data_samples.append(pose_data_sample)

    return results, data_samples
