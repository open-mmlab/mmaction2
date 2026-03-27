import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import mmcv
from mmaction.apis.enhanced_inference import enhanced_pose_inference
from mmdet.apis import inference_detector, init_detector


def main():
    # Detector settings
    det_config = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py'
    det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

    # Pose estimation settings
    pose_config = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
    pose_checkpoint = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/best_acc_top1_epoch_24.pth'

    # Initialize detector
    det_model = init_detector(det_config, det_checkpoint, device='cuda:0')

    # Input video
    video_path = '/home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/data/skeleton/Le2i/Lecture_room/video_1.avi'

    # Extract frames
    video = mmcv.VideoReader(video_path)
    frames = [video[i] for i in range(len(video))]
    frame_paths = [f'tmp_frame_{i}.jpg' for i in range(len(frames))]

    # Save frames temporarily
    for frame, path in zip(frames, frame_paths):
        mmcv.imwrite(frame, path)

    # Run human detection on each frame
    det_results = []
    for frame_path in frame_paths:
        result = inference_detector(det_model, frame_path)
        # Keep only person class (usually class 0)
        det_results.append(
            result.pred_instances.bboxes[result.pred_instances.labels == 0].cpu().numpy())

    # Enhanced pose estimation with batching and temporal smoothing
    pose_results, _ = enhanced_pose_inference(
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        frame_paths=frame_paths,
        det_results=det_results,
        device='cuda:0',
        batch_size=4,  # Process 4 frames at once
        use_temporal_smoothing=True,  # Apply smoothing for video
        smoothing_window_size=7,
        smoothing_sigma=1.5,
        keypoint_threshold=0.3
    )

    # Clean up temporary files
    for path in frame_paths:
        if os.path.exists(path):
            os.remove(path)

    # Now you can use pose_results for further processing,
    # such as skeleton-based action recognition
    print(f"Successfully processed {len(pose_results)} frames")


if __name__ == '__main__':
    main()
