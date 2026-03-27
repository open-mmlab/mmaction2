# Copyright (c) OpenMMLab. All rights reserved.
"""
NTU RGB+D Pose Extraction Module

This module contains functions to extract human pose keypoints from videos,
particularly focused on the NTU RGB+D dataset format. It provides a complete
pipeline for:
1. Human detection in video frames
2. Post-processing of detection results with tracklet formation
3. Pose estimation based on detections
4. Aligning pose keypoints across frames
5. Creating standardized annotation format for action recognition

The implementation handles both single-person and multi-person scenarios
with special consideration for challenging detection cases.
"""

import abc
import argparse
import os.path as osp
from collections import defaultdict
from tempfile import TemporaryDirectory

import mmengine
import numpy as np
import torch

from mmaction.apis import detection_inference, pose_inference
from mmaction.utils import frame_extract


class Args:
    """
    Configuration parameters for pose extraction pipeline.

    This class holds default values for detection and pose estimation
    models, confidence thresholds, and device configuration.
    """

    def __init__(self):
        # Human detection model settings
        self.det_config = 'demo/demo_configs/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
        self.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
        self.det_score_thr = 0.5  # Detection confidence threshold

        # Pose estimation model settings
        self.pose_config = 'demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py'
        self.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
        
        # https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth

        # General settings
        self.device = 'cuda:0'
        self.skip_postproc = False


# Create a global instance of Args with default values
args = Args()


def intersection(b0, b1):
    """
    Calculate intersection area of two bounding boxes.

    Args:
        b0 (ndarray): First bounding box in format [x1, y1, x2, y2, ...]
        b1 (ndarray): Second bounding box in format [x1, y1, x2, y2, ...]

    Returns:
        float: Intersection area
    """
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        b0 (ndarray): First bounding box in format [x1, y1, x2, y2, ...]
        b1 (ndarray): Second bounding box in format [x1, y1, x2, y2, ...]

    Returns:
        float: IoU value between 0 and 1
    """
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    """
    Calculate area of a bounding box.

    Args:
        b (ndarray): Bounding box in format [x1, y1, x2, y2, ...]

    Returns:
        float: Area of the bounding box
    """
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):
    """
    Remove duplicate or heavily overlapping bounding boxes.

    Keeps boxes with higher confidence scores when significant overlap exists.

    Args:
        bbox (ndarray): Array of bounding boxes, each in format [x1, y1, x2, y2, score]

    Returns:
        ndarray: Filtered array of bounding boxes
    """
    def inside(box0, box1, threshold=0.8):
        """Check if box0 is mostly inside box1."""
        return intersection(box0, box1) / area(box0) > threshold

    num_bboxes = bbox.shape[0]
    if num_bboxes <= 1:
        return bbox

    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            # If box i is inside box j and has lower confidence, remove box i
            if i != j and inside(bbox[i], bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]


def is_easy_example(det_results, num_person):
    """
    Determine if a video has consistent high-confidence detections.

    An "easy example" has exactly the expected number of persons
    with high confidence (>0.95) detections across all frames.

    Args:
        det_results (list): List of detection results per frame
        num_person (int): Expected number of persons (1 or 2)

    Returns:
        tuple: (is_easy, bboxes), where:
            - is_easy (bool): Whether this is an easy example
            - bboxes (ndarray or int): If easy, stacked bounding boxes; 
                                      otherwise, the number of high-confidence detections
    """
    threshold = 0.95

    def thre_bbox(bboxes, threshold=threshold):
        """Count and verify consistent high-confidence detections."""
        shape = [sum(bbox[:, -1] > threshold) for bbox in bboxes]
        ret = np.all(np.array(shape) == shape[0])
        return shape[0] if ret else -1

    if thre_bbox(det_results) == num_person:
        # This is an easy example - filter to keep only high-confidence detections
        det_results = [x[x[..., -1] > 0.95] for x in det_results]
        return True, np.stack(det_results)
    return False, thre_bbox(det_results)


def bbox2tracklet(bbox):
    """
    Convert frame-by-frame bounding boxes to consistent tracklets.

    A tracklet is a sequence of bounding boxes that follows the same person
    across multiple frames. This function uses IoU to associate detections
    across frames.

    Args:
        bbox (list): List of bounding box arrays per frame

    Returns:
        dict: Dictionary mapping tracklet IDs to lists of (frame_idx, bbox) tuples
    """
    iou_thre = 0.6  # IoU threshold for tracklet association
    tracklet_id = -1
    tracklet_st_frame = {}  # Start frame for each tracklet
    tracklets = defaultdict(list)

    # Process each frame and each detection
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            # Try to match with existing tracklets (from newest to oldest)
            for tlet_id in range(tracklet_id, -1, -1):
                # Conditions for matching:
                # 1. IoU with latest box in tracklet exceeds threshold
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                # 2. Not too far apart in time (max 10 frame gap)
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) < 10)
                # 3. Current frame not already in this tracklet
                cond3 = tracklets[tlet_id][-1][0] != t

                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break

            # If no match found, create a new tracklet
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))

    return tracklets


def drop_tracklet(tracklet):
    """
    Filter out short or small tracklets.

    Removes tracklets that:
    1. Have fewer than 6 frames
    2. Have a mean bounding box area less than 5000 pixels

    Args:
        tracklet (dict): Dictionary of tracklets from bbox2tracklet

    Returns:
        dict: Filtered dictionary of tracklets
    """
    # Filter out short tracklets (less than 6 frames)
    tracklet = {k: v for k, v in tracklet.items() if len(v) > 5}

    def meanarea(track):
        """Calculate mean area of bounding boxes in a tracklet."""
        boxes = np.stack([x[1] for x in track]).astype(np.float32)
        areas = (boxes[..., 2] - boxes[..., 0]) * \
            (boxes[..., 3] - boxes[..., 1])
        return np.mean(areas)

    # Filter out small tracklets (mean area < 5000 pixels)
    tracklet = {k: v for k, v in tracklet.items() if meanarea(v) > 5000}
    return tracklet


def distance_tracklet(tracklet):
    """
    Calculate mean distance of each tracklet from the center of the frame.

    This is used to prioritize tracklets that are closer to the center
    when there are multiple candidates.

    Args:
        tracklet (dict): Dictionary of tracklets

    Returns:
        dict: Dictionary mapping tracklet IDs to mean distances from center
    """
    dists = {}
    for k, v in tracklet.items():
        # Stack all bounding boxes in this tracklet
        bboxes = np.stack([x[1] for x in v])

        # Calculate center coordinates of each bbox
        c_x = (bboxes[..., 2] + bboxes[..., 0]) / 2.
        c_y = (bboxes[..., 3] + bboxes[..., 1]) / 2.

        # Adjust to center relative to frame center (assumed to be 480, 270)
        # This is an approximation based on the original NTU dataset
        c_x -= 480
        c_y -= 270

        # Calculate distance from center
        c = np.concatenate([c_x[..., None], c_y[..., None]], axis=1)
        dist = np.linalg.norm(c, axis=1)
        dists[k] = np.mean(dist)

    return dists


def tracklet2bbox(track, num_frame):
    """
    Convert a single tracklet to frame-by-frame bounding boxes.

    For frames where the tracklet has no detection, interpolates
    from nearest available frame.

    Args:
        track (list): List of (frame_idx, bbox) tuples for a single tracklet
        num_frame (int): Total number of frames in the video

    Returns:
        ndarray: Array of bounding boxes, one per frame
    """
    # Initialize empty bounding boxes for all frames
    bbox = np.zeros((num_frame, 5))
    trackd = {}

    # Populate with known detections
    for k, v in track:
        bbox[k] = v
        trackd[k] = v

    # Fill in missing frames by finding nearest detection
    for i in range(num_frame):
        if bbox[i][-1] <= 0.5:  # Low-confidence or missing detection
            mind = np.Inf
            nearest_idx = None

            # Find nearest valid frame
            for k in trackd:
                if np.abs(k - i) < mind:
                    mind = np.abs(k - i)
                    nearest_idx = k

            # Copy detection from nearest frame
            bbox[i] = bbox[nearest_idx]

    return bbox


def tracklets2bbox(tracklet, num_frame):
    """
    Convert multiple tracklets to a single primary tracklet with bounding boxes.

    Prioritizes tracklets that:
    1. Cover at least half the video length
    2. Are closer to the center of the frame

    Args:
        tracklet (dict): Dictionary of tracklets
        num_frame (int): Total number of frames in the video

    Returns:
        tuple: (bad_frames, bboxes), where:
            - bad_frames (int): Number of frames with low-confidence detections
            - bboxes (ndarray): Array of bounding boxes with shape (num_frame, 1, 5)
    """
    # Calculate distances from center for all tracklets
    dists = distance_tracklet(tracklet)
    sorted_inds = sorted(dists, key=lambda x: dists[x])

    # Find a long enough tracklet (covers at least half the frames)
    # and set a distance threshold based on it
    dist_thre = np.Inf
    for i in sorted_inds:
        if len(tracklet[i]) >= num_frame / 2:
            # Use 2x the distance of this tracklet as threshold
            dist_thre = 2 * dists[i]
            break

    # Set a minimum distance threshold
    dist_thre = max(50, dist_thre)

    # Initialize empty bounding boxes
    bbox = np.zeros((num_frame, 5))
    bboxd = {}

    # Fill in detections from tracklets within distance threshold
    for idx in sorted_inds:
        if dists[idx] < dist_thre:
            for k, v in tracklet[idx]:
                if bbox[k][-1] < 0.01:  # Empty or very low confidence
                    bbox[k] = v
                    bboxd[k] = v

    # Count and fix frames with missing detections
    bad = 0
    for idx in range(num_frame):
        if bbox[idx][-1] < 0.01:
            bad += 1

            # Find nearest frame with detection
            mind = np.Inf
            mink = None
            for k in bboxd:
                if np.abs(k - idx) < mind:
                    mind = np.abs(k - idx)
                    mink = k

            # Copy detection from nearest frame
            bbox[idx] = bboxd[mink]

    # Reshape to match expected format (num_frame, 1, 5)
    return bad, bbox[:, None, :]


def bboxes2bbox(bbox, num_frame):
    """
    Process bounding boxes for two-person scenario.

    For each frame, select the top 2 detections by confidence score,
    and maintain consistent person IDs across frames using IoU.

    Args:
        bbox (list): List of bounding box arrays per frame
        num_frame (int): Total number of frames in the video

    Returns:
        ndarray: Array of shape (num_frame, 2, 5) containing consistent
                two-person tracking results
    """
    # Initialize array for two persons across all frames
    ret = np.zeros((num_frame, 2, 5))

    # Process each frame's detections
    for t, item in enumerate(bbox):
        if item.shape[0] <= 2:
            # If we have 0, 1 or 2 detections, use them as is
            ret[t, :item.shape[0]] = item
        else:
            # If we have more than 2, select top 2 by confidence
            inds = sorted(
                list(range(item.shape[0])), key=lambda x: -item[x, -1])
            ret[t] = item[inds[:2]]

    # Process frames to maintain consistent person IDs
    for t in range(num_frame):
        # Handle frames with no detections
        if ret[t, 0, -1] <= 0.01:
            ret[t] = ret[t - 1]  # Copy from previous frame

        # Handle frames with only one detection (need to decide which person it is)
        elif ret[t, 1, -1] <= 0.01 and t > 0:
            if ret[t - 1, 0, -1] > 0.01 and ret[t - 1, 1, -1] > 0.01:
                # Determine which previous person this detection matches better
                if iou(ret[t, 0], ret[t - 1, 0]) > iou(ret[t, 0], ret[t - 1, 1]):
                    # Detection matches person 0, so copy person 1 from previous frame
                    ret[t, 1] = ret[t - 1, 1]
                else:
                    # Detection matches person 1, so copy person 0 and swap positions
                    ret[t, 1] = ret[t, 0]
                    ret[t, 0] = ret[t - 1, 0]

    return ret


def ntu_det_postproc(vid, det_results):
    """
    Post-process detection results for NTU RGB+D format videos.

    This is the main algorithm for handling both easy and hard detection cases,
    determining number of persons, and ensuring consistent tracking.

    Args:
        vid (str): Path to video file
        det_results (list): List of detection results per frame

    Returns:
        ndarray: Processed detection results with consistent tracking
    """
    # Remove duplicate detections in each frame
    det_results = [removedup(x) for x in det_results]

    # Determine number of persons from video filename
    # NTU RGB+D follows a specific naming convention
    try:
        label = int(vid.split('/')[-1].split('A')[1][:3])
        # Actions 50-60 and 106-120 in NTU RGB+D are multi-person actions
        mpaction = list(range(50, 61)) + list(range(106, 121))
        n_person = 2 if label in mpaction else 1
    except (IndexError, ValueError):
        # For videos not following the convention, assume single person
        n_person = 1

    # Check if this is an easy example (consistent high-confidence detections)
    is_easy, bboxes = is_easy_example(det_results, n_person)
    if is_easy:
        print('\nEasy Example')
        return bboxes

    # For harder cases, create tracklets from detections
    tracklets = bbox2tracklet(det_results)
    tracklets = drop_tracklet(tracklets)

    print(f'\nHard {n_person}-person Example, found {len(tracklets)} tracklet')

    # Handle single-person case
    if n_person == 1:
        if len(tracklets) == 1:
            # Only one tracklet found, use it directly
            tracklet = list(tracklets.values())[0]
            det_results = tracklet2bbox(tracklet, len(det_results))
            return np.stack(det_results)
        else:
            # Multiple tracklets found, select the best one
            bad, det_results = tracklets2bbox(tracklets, len(det_results))
            return det_results

    # Handle two-person case
    if len(tracklets) <= 2:
        # We found exactly the right number of tracklets
        tracklets = list(tracklets.values())
        bboxes = []
        for tracklet in tracklets:
            bboxes.append(tracklet2bbox(tracklet, len(det_results))[:, None])
        bbox = np.concatenate(bboxes, axis=1)
        return bbox
    else:
        # Too many tracklets, need to select and organize them
        return bboxes2bbox(det_results, len(det_results))


def pose_inference_with_align(frame_paths, det_results, device):
    """
    Perform pose estimation on detection results and align across frames.

    This function:
    1. Runs pose estimation on each detected person
    2. Aligns pose data across frames (handling variable person counts)
    3. Formats keypoints and confidence scores for each joint

    Args:
        frame_paths (list): List of paths to video frames
        det_results (list): List of detection results per frame
        device (str): Device to run inference on ('cuda:0', 'cpu', etc.)

    Returns:
        tuple: (keypoints, scores), where:
            - keypoints (ndarray): Array of shape (num_persons, num_frames, num_joints, 2)
            - scores (ndarray): Array of shape (num_persons, num_frames, num_joints)
    """
    # Set up args for pose inference
    pose_args = Args()
    pose_args.device = device

    # Filter out frames without any detections
    det_results = [
        frm_dets for frm_dets in det_results if frm_dets.shape[0] > 0]

    # Run pose estimation
    pose_results, _ = pose_inference(
        pose_args.pose_config,
        pose_args.pose_checkpoint,
        frame_paths,
        det_results,
        device
    )

    # Align the pose results to have consistent num_person across frames
    # Find the maximum number of persons detected in any frame
    num_persons = max([pose['keypoints'].shape[0] for pose in pose_results])
    num_points = pose_results[0]['keypoints'].shape[1]
    num_frames = len(pose_results)

    # Initialize arrays for aligned pose data
    keypoints = np.zeros(
        (num_persons, num_frames, num_points, 2), dtype=np.float32)
    scores = np.zeros((num_persons, num_frames, num_points), dtype=np.float32)

    # Copy pose data for each person and frame
    for f_idx, frm_pose in enumerate(pose_results):
        frm_num_persons = frm_pose['keypoints'].shape[0]
        for p_idx in range(frm_num_persons):
            keypoints[p_idx, f_idx] = frm_pose['keypoints'][p_idx]
            scores[p_idx, f_idx] = frm_pose['keypoint_scores'][p_idx]

    return keypoints, scores


def ntu_pose_extraction(vid, skip_postproc=False, device='cuda:0'):
    """
    Extract pose keypoints from a video for action recognition.

    Complete pipeline that:
    1. Extracts frames from video
    2. Detects persons in frames
    3. Processes detections for consistent tracking
    4. Estimates pose keypoints
    5. Formats results into a structured annotation

    Args:
        vid (str): Path to video file
        skip_postproc (bool): Whether to skip detection post-processing
        device (str): Device to run inference on ('cuda:0', 'cpu', etc.)

    Returns:
        dict: Annotation dictionary containing:
            - keypoint: Pose keypoints array (num_persons, num_frames, num_joints, 2)
            - keypoint_score: Confidence scores (num_persons, num_frames, num_joints)
            - frame_dir: Video name without extension
            - img_shape: Image dimensions (height, width)
            - original_shape: Original image dimensions
            - total_frames: Number of frames
            - label: Action label (if filename follows NTU format)
    """
    # Create a temporary directory for extracted frames
    tmp_dir = TemporaryDirectory()
    frame_paths, _ = frame_extract(vid, out_dir=tmp_dir.name)

    # Set up detection args
    det_args = Args()
    det_args.device = device

    # Run human detection
    det_results, _ = detection_inference(
        det_args.det_config,
        det_args.det_checkpoint,
        frame_paths,
        det_args.det_score_thr,
        device=device,
        with_score=True
    )

    # Post-process detection results for consistent tracking
    if not skip_postproc:
        det_results = ntu_det_postproc(vid, det_results)

    # Create annotation dictionary
    anno = dict()

    # Extract pose keypoints and scores
    keypoints, scores = pose_inference_with_align(
        frame_paths, det_results, device)

    # Fill annotation dictionary
    anno['keypoint'] = keypoints
    anno['keypoint_score'] = scores
    anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    # Default shape, will be used if not detected
    anno['img_shape'] = (1080, 1920)
    anno['original_shape'] = (1080, 1920)
    anno['total_frames'] = keypoints.shape[1]

    # Extract label from filename if it follows NTU naming convention
    try:
        # Original NTU dataset filename format: S001C001P001R001A001.avi
        # where A001 is action class 1
        anno['label'] = int(osp.basename(vid).split('A')[1][:3]) - 1
    except (IndexError, ValueError):
        # For other video formats, set a default label
        print(
            f"Warning: Video filename '{osp.basename(vid)}' doesn't follow NTU naming convention.")
        print("Setting default label to 0.")
        anno['label'] = 0

    # Clean up temporary directory
    tmp_dir.cleanup()
    return anno


def parse_args():
    """
    Parse command line arguments for standalone script usage.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('output', type=str, help='output pickle name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for inference, e.g., cuda:0, cpu')
    parser.add_argument('--skip-postproc', action='store_true',
                        help='Skip detection post-processing (tracking)')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse command line arguments
    cmd_args = parse_args()

    # Extract pose annotations
    anno = ntu_pose_extraction(
        cmd_args.video,
        cmd_args.skip_postproc,
        cmd_args.device
    )

    # Save the annotations
    mmengine.dump(anno, cmd_args.output)
