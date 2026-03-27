import torch
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
from pathlib import Path
import torch.nn as nn
import mmengine
from mmengine.utils import track_iter_progress


def enhanced_pose_inference(
    pose_config: Union[str, Path, mmengine.Config, nn.Module],
    pose_checkpoint: str,
    frame_paths: List[str],
    det_results: List[np.ndarray],
    device: Union[str, torch.device] = 'cuda:0',
    batch_size: int = 1,
    use_temporal_smoothing: bool = False,
    smoothing_window_size: int = 7,
    smoothing_sigma: float = 1.5,
    keypoint_threshold: float = 0.3,
    return_heatmaps: bool = False
) -> Tuple[List[Dict[str, np.ndarray]], List]:
    """Enhanced pose estimation with batching and temporal smoothing options.

    Args:
        pose_config: Pose config file path or model object.
        pose_checkpoint: Checkpoint path/url.
        frame_paths: The paths of frames to do pose inference.
        det_results: List of detected human boxes.
        device: The desired device for inference.
        batch_size: Number of frames to process in a batch.
        use_temporal_smoothing: Whether to apply temporal smoothing to keypoints.
        smoothing_window_size: Size of the Gaussian smoothing window.
        smoothing_sigma: Standard deviation for Gaussian kernel.
        keypoint_threshold: Confidence threshold for keypoints.
        return_heatmaps: Whether to return heatmaps along with keypoints.

    Returns:
        Tuple of pose estimation results and data samples.
    """
    try:
        from mmpose.apis import inference_topdown, init_model
        from mmpose.structures import PoseDataSample, merge_data_samples
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import required modules from MMPose')

    # Import the temporal_smoothing function we implemented above
    try:
        from mmaction.apis.temporal_processing import apply_temporal_smoothing
    except ImportError:
        use_temporal_smoothing = False
        print("Warning: Temporal smoothing module not found. Disabling smoothing.")

    # Input validation
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty")
    if len(frame_paths) != len(det_results):
        raise ValueError(f"Number of frames ({len(frame_paths)}) must match "
                         f"number of detection results ({len(det_results)})")

    # Model initialization
    if isinstance(pose_config, nn.Module):
        model = pose_config
    else:
        model = init_model(pose_config, pose_checkpoint, device)

    # Get the correct number of keypoints from model metadata
    try:
        num_keypoints = model.dataset_meta['num_keypoints']
    except (KeyError, AttributeError):
        # For action recognition models like PoseC3D
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'in_channels'):
            # If it's a pose recognition model, use in_channels
            num_keypoints = model.backbone.in_channels
            print(
                f"Using backbone in_channels as num_keypoints: {num_keypoints}")
        else:
            # Default to standard COCO keypoints
            print("Warning: Could not determine number of keypoints, using default (17)")
            num_keypoints = 17

    results = []
    data_samples = []

    # Process frames (with optional batching)
    if batch_size <= 1:
        # Single frame processing
        print('Performing Human Pose Estimation for each frame')
        for f, d in track_iter_progress(list(zip(frame_paths, det_results))):
            # Validate detection format
            if d.size > 0 and d.shape[1] < 4:
                raise ValueError(
                    f"Detection boxes must have at least 4 values (x1,y1,x2,y2), got shape {d.shape}")

            pose_data_samples: List[PoseDataSample] = inference_topdown(
                model, f, d[..., :4], bbox_format='xyxy')
            pose_data_sample = merge_data_samples(pose_data_samples)
            pose_data_sample.dataset_meta = model.dataset_meta

            # Handle empty predictions
            if not hasattr(pose_data_sample, 'pred_instances'):
                pred_instances_data = dict(
                    keypoints=np.empty(shape=(0, num_keypoints, 2)),
                    keypoints_scores=np.empty(
                        shape=(0, num_keypoints), dtype=np.float32),
                    bboxes=np.empty(shape=(0, 4), dtype=np.float32),
                    bbox_scores=np.empty(shape=(0), dtype=np.float32))
                pose_data_sample.pred_instances = InstanceData(
                    **pred_instances_data)

            poses = pose_data_sample.pred_instances.to_dict()

            # Filter low-confidence keypoints
            if 'keypoints_scores' in poses and poses['keypoints_scores'].size > 0:
                mask = poses['keypoints_scores'] < keypoint_threshold
                # Set low-confidence keypoints to 0
                if 'keypoints' in poses and poses['keypoints'].size > 0:
                    poses['keypoints'][mask] = 0

            results.append(poses)
            data_samples.append(pose_data_sample)
    else:
        # Use the batch processing function we implemented above
        try:
            from mmaction.apis.batch_inference import batch_pose_inference
            results, data_samples = batch_pose_inference(
                pose_config, pose_checkpoint, frame_paths, det_results,
                batch_size=batch_size, device=device
            )
        except ImportError:
            print(
                "Warning: Batch inference module not found. Falling back to single frame processing.")
            # Recursively call this function with batch_size=1
            return enhanced_pose_inference(
                pose_config, pose_checkpoint, frame_paths, det_results,
                device=device, batch_size=1, use_temporal_smoothing=use_temporal_smoothing,
                smoothing_window_size=smoothing_window_size, smoothing_sigma=smoothing_sigma,
                keypoint_threshold=keypoint_threshold, return_heatmaps=return_heatmaps
            )

    # Apply temporal smoothing if requested
    if use_temporal_smoothing and len(results) > 3:
        results = apply_temporal_smoothing(
            results,
            window_size=smoothing_window_size,
            sigma=smoothing_sigma,
            min_score_threshold=keypoint_threshold
        )

    return results, data_samples
