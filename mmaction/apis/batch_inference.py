import torch
import numpy as np
from typing import List, Union, Tuple, Dict
from pathlib import Path
import torch.nn as nn
import mmengine
from tqdm import tqdm


def batch_pose_inference(
    pose_config: Union[str, Path, mmengine.Config, nn.Module],
    pose_checkpoint: str,
    frame_paths: List[str],
    det_results: List[np.ndarray],
    batch_size: int = 4,
    device: Union[str, torch.device] = 'cuda:0'
) -> Tuple[List[Dict[str, np.ndarray]], List]:
    """Perform batched Top-Down pose estimation for better throughput.

    Args:
        pose_config: Pose config file path or model object.
        pose_checkpoint: Checkpoint path/url.
        frame_paths: The paths of frames to do pose inference.
        det_results: List of detected human boxes.
        batch_size: Number of frames to process in a batch.
        device: The desired device for inference.

    Returns:
        Tuple of pose estimation results and data samples.
    """
    try:
        from mmpose.apis import init_model
        from mmpose.structures import PoseDataSample, merge_data_samples
        import mmcv
    except (ImportError, ModuleNotFoundError):
        raise ImportError('Failed to import required modules from MMPose')

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
    num_keypoints = model.dataset_meta['num_keypoints']

    results = []
    data_samples = []

    # Process in batches
    total_batches = (len(frame_paths) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frame_paths))

        batch_frame_paths = frame_paths[start_idx:end_idx]
        batch_det_results = det_results[start_idx:end_idx]

        # Load images
        batch_images = [mmcv.imread(frame_path)
                        for frame_path in batch_frame_paths]

        # Process each image in the batch
        for img, dets, frame_path in zip(batch_images, batch_det_results, batch_frame_paths):
            # Validate detection format
            if dets.size > 0 and dets.shape[1] < 4:
                raise ValueError(
                    f"Detection boxes must have at least 4 values (x1,y1,x2,y2), got shape {dets.shape}")

            # Run inference using the model's test_step directly with image data
            # This avoids repeated I/O operations and is more efficient
            data_info = dict(img=img, bbox=dets[..., :4])
            data = model.data_preprocessor(data_info, False)

            with torch.no_grad():
                predictions = model.forward(data, mode='predict')

            pose_data_sample = merge_data_samples(predictions)
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
            results.append(poses)
            data_samples.append(pose_data_sample)

    return results, data_samples
