"""
Copyright (c) OpenMMLab. All rights reserved.
Refactored skeleton-based action recognition demo with improved organization.
"""
import argparse
import tempfile
from typing import List, Tuple, Dict

import cv2
import mmcv
import mmengine
import torch
from mmengine import DictAction
from mmengine.utils import track_iter_progress

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

from mmaction.apis import (
    detection_inference,
    inference_skeleton,
    init_recognizer,
    pose_inference
)
from mmaction.utils import frame_extract
from mmaction.registry import VISUALIZERS

# Visualization settings
VISUALIZATION_SETTINGS = {
    'font_face': cv2.FONT_HERSHEY_DUPLEX,
    'font_scale': 0.75,
    'font_color': (255, 0, 0),  # BGR, white
    'thickness': 1,
    'line_type': 1
}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='MMAction2 skeleton-based demo')

    # Input and output
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')

    # Model configs
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'),
        help='skeleton model checkpoint file/url')

    # Detection configs
    parser.add_argument(
        '--det-config',
        default='demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='human detection score threshold')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='human category id for detection')

    # Pose estimation configs
    parser.add_argument(
        '--pose-config',
        default='demo/demo_configs/'
        'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')

    # Other settings
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu60.txt',
        help='label map file')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--window-size',
        type=int,
        default=32,
        help='window size for skeleton action recognition')
    parser.add_argument(
        '--window-stride',
        type=int,
        default=16,
        help='stride for sliding window')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config')

    return parser.parse_args()


def process_frame_window(
    model: torch.nn.Module,
    pose_results: List[Dict],
    img_shape: Tuple[int, int],
    label_map: List[str]
) -> Tuple[str, float, List[float]]:
    """Process a window of frames for action recognition.

    Args:
        model: The loaded recognizer model.
        pose_results: List of pose estimation results.
        img_shape: Original image shape.
        label_map: List of action labels.

    Returns:
        Tuple containing action label, confidence score, and all prediction scores.
    """
    result = inference_skeleton(model, pose_results, img_shape)

    pred_scores = result.pred_score.cpu().numpy()
    max_pred_index = pred_scores.argmax()

    action_label = label_map[max_pred_index]
    confidence = pred_scores[max_pred_index]

    return action_label, confidence, pred_scores


def check_falling_alert(action_label: str, confidence: float, frame_range: Tuple[int, int]):
    """Generate warning if falling is detected with confidence threshold."""
    if action_label.lower() == 'falling' and confidence > 0.7:
        start, end = frame_range
        warning_msg = f"⚠️ FALL DETECTED! ⚠️\nConfidence: {confidence:.2%}\nFrames: {start}-{end}"
        print("\n" + "!"*50)
        print(warning_msg)
        print("!"*50 + "\n")


def visualize_with_labels(
    args,
    frames: List[torch.Tensor],
    pose_data_samples: List,
    frame_labels: List[str],
    frame_confidences: List[float]
) -> None:
    """Visualize frames with action labels and skeleton overlays.

    Args:
        args: Parsed command line arguments.
        frames: List of video frames.
        pose_data_samples: List of pose estimation results.
        frame_labels: List of action labels for each frame.
        frame_confidences: List of confidence scores for each frame.
    """
    pose_config = mmengine.Config.fromfile(args.pose_config)
    visualizer = VISUALIZERS.build(pose_config.visualizer)
    visualizer.set_dataset_meta(pose_data_samples[0].dataset_meta)

    vis_frames = []
    print('Drawing skeleton and labels for each frame')

    for i, (d, f) in enumerate(track_iter_progress(list(zip(pose_data_samples, frames)))):
        # Convert frame color space
        f = mmcv.imconvert(f, 'bgr', 'rgb')

        # Draw pose estimation results
        visualizer.add_datasample(
            'result',
            f,
            data_sample=d,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=None,
            kpt_thr=0.3
        )
        vis_frame = visualizer.get_image()

        # Add action label and confidence
        action_text = f"{frame_labels[i]} ({frame_confidences[i]:.2f})"

        # Highlight falling labels in red
        if "falling" in frame_labels[i].lower():
            FONTCOLOR = (255, 0, 0)  # Red color for falling
            thickness = 2
        else:
            FONTCOLOR = VISUALIZATION_SETTINGS['font_color']
            thickness = VISUALIZATION_SETTINGS['thickness']

        cv2.putText(
            vis_frame,
            action_text,
            (10, 30),
            VISUALIZATION_SETTINGS['font_face'],
            VISUALIZATION_SETTINGS['font_scale'],
            FONTCOLOR,  # Now using conditional color
            thickness,  # Now using conditional thickness
            VISUALIZATION_SETTINGS['line_type']
        )

        # Add frame number
        cv2.putText(
            vis_frame,
            f"Frame: {i}",
            (10, 60),
            VISUALIZATION_SETTINGS['font_face'],
            VISUALIZATION_SETTINGS['font_scale'],
            VISUALIZATION_SETTINGS['font_color'],
            VISUALIZATION_SETTINGS['thickness'],
            VISUALIZATION_SETTINGS['line_type']
        )

        vis_frames.append(vis_frame)

    # Create and save video
    vid = mpy.ImageSequenceClip(vis_frames, fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)


def process_video_windows(
    model: torch.nn.Module,
    pose_results: List[Dict],
    img_shape: Tuple[int, int],
    label_map: List[str],
    window_size: int,
    window_stride: int,
    num_frames: int
) -> Tuple[List[str], List[float]]:
    """Process video using sliding windows for action recognition.

    Args:
        model: The loaded recognizer model.
        pose_results: List of pose estimation results.
        img_shape: Original image shape.
        label_map: List of action labels.
        window_size: Size of sliding window.
        window_stride: Stride for sliding window.
        num_frames: Total number of frames.

    Returns:
        Tuple containing lists of frame labels and confidence scores.
    """
    frame_labels = ["Unknown"] * num_frames
    frame_confidences = [0.0] * num_frames

    print('Processing video in sliding windows')
    for start_idx in track_iter_progress(range(0, num_frames - window_size + 1, window_stride)):
        end_idx = start_idx + window_size
        window_pose_results = pose_results[start_idx:end_idx]

        action_label, confidence, _ = process_frame_window(
            model, window_pose_results, img_shape, label_map)

        # Add falling detection check
        check_falling_alert(action_label, confidence, (start_idx, end_idx))

        # Assign labels to frames in the window based on confidence
        for i in range(start_idx, end_idx):
            if confidence > frame_confidences[i]:
                frame_labels[i] = action_label
                frame_confidences[i] = confidence

    return frame_labels, frame_confidences


def main():
    """Main function for skeleton-based action recognition demo."""
    args = parse_args()

    # Create temporary directory for frame extraction
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Extract video frames
        frame_paths, frames = frame_extract(
            args.video, args.short_side, tmp_dir)
        h, w, _ = frames[0].shape

        # Perform human detection
        det_results, _ = detection_inference(
            args.det_config,
            args.det_checkpoint,
            frame_paths,
            args.det_score_thr,
            args.det_cat_id,
            args.device
        )
        torch.cuda.empty_cache()

        # Perform pose estimation
        pose_results, pose_data_samples = pose_inference(
            args.pose_config,
            args.pose_checkpoint,
            frame_paths,
            det_results,
            args.device
        )
        torch.cuda.empty_cache()

        # Initialize action recognition model
        config = mmengine.Config.fromfile(args.config)
        config.merge_from_dict(args.cfg_options)
        model = init_recognizer(config, args.checkpoint, args.device)

        # Load action labels
        label_map = [x.strip() for x in open(args.label_map).readlines()]

        # Process video with sliding windows
        frame_labels, frame_confidences = process_video_windows(
            model,
            pose_results,
            (h, w),
            label_map,
            args.window_size,
            args.window_stride,
            len(pose_results)
        )

        # Visualize results
        visualize_with_labels(
            args,
            frames,
            pose_data_samples,
            frame_labels,
            frame_confidences
        )


if __name__ == '__main__':
    main()
