#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import pickle
import numpy as np
import cv2
import os
import os.path as osp

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

# Define the skeleton connections (edges between joints)
# Based on the standard 17 keypoints (COCO format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

# Define colors for visualization
KEYPOINT_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (0, 0, 255)  # Red
THICKNESS = 2
KEYPOINT_RADIUS = 4


def parse_args():
    parser = argparse.ArgumentParser(description='Overlay skeleton on video')
    parser.add_argument('video_file', help='path to the original video file')
    parser.add_argument('pickle_file', help='path to the skeleton pickle file')
    parser.add_argument('out_filename', help='output video filename')
    parser.add_argument('--keypoint_threshold', type=float, default=0.3,
                        help='Threshold for keypoint confidence score')
    parser.add_argument('--fps', type=int, default=30,
                        help='FPS for output video')
    args = parser.parse_args()
    return args


def overlay_skeleton_on_video(video_file, pickle_file, out_filename, keypoint_threshold=0.3, fps=30):
    # Load skeleton data
    print(f"Loading skeleton data from {pickle_file}")
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Extract keypoints and scores
    # shape: (num_person, num_frames, num_keypoints, 2)
    keypoints = data['keypoint']
    # shape: (num_person, num_frames, num_keypoints)
    keypoint_scores = data['keypoint_score']
    num_persons, num_frames, num_keypoints, _ = keypoints.shape

    print(
        f"Loaded data with {num_persons} persons, {num_frames} frames, {num_keypoints} keypoints")

    # Open the video file
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_file}")

    # Get video properties
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Video properties: {video_width}x{video_height}, {video_fps} fps, {total_frames} frames")

    # Create a temporary directory for frames
    tmp_dir = "tmp_overlay_vis"
    os.makedirs(tmp_dir, exist_ok=True)

    frame_files = []
    frame_idx = 0

    # Create temp directory for output frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if there are more video frames than skeleton frames
        if frame_idx >= num_frames:
            break

        # Draw skeletons for each person on this frame
        for person_idx in range(num_persons):
            # Get skeleton data for this person in this frame
            person_keypoints = keypoints[person_idx, frame_idx]
            person_scores = keypoint_scores[person_idx, frame_idx]

            # Draw keypoints
            for kp_idx in range(num_keypoints):
                x, y = person_keypoints[kp_idx]
                score = person_scores[kp_idx]

                # Only draw keypoints with score above threshold
                if score > keypoint_threshold:
                    x, y = int(x), int(y)
                    cv2.circle(frame, (x, y), KEYPOINT_RADIUS,
                               KEYPOINT_COLOR, -1)

            # Draw skeleton connections
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                start_score = person_scores[start_idx]
                end_score = person_scores[end_idx]

                # Only draw connection if both keypoints have score above threshold
                if start_score > keypoint_threshold and end_score > keypoint_threshold:
                    start_x, start_y = person_keypoints[start_idx]
                    end_x, end_y = person_keypoints[end_idx]

                    start_x, start_y = int(start_x), int(start_y)
                    end_x, end_y = int(end_x), int(end_y)

                    cv2.line(frame, (start_x, start_y),
                             (end_x, end_y), CONNECTION_COLOR, THICKNESS)

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Save the frame
        frame_file = osp.join(tmp_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_file, frame)
        frame_files.append(frame_file)

        frame_idx += 1

        # Print progress every 10%
        if frame_idx % (num_frames // 10) == 0 or frame_idx == 1:
            print(f"Processed {frame_idx}/{num_frames} frames")

    cap.release()

    # Create a video from the frames
    print(f"Creating video from {len(frame_files)} frames")
    clip = mpy.ImageSequenceClip(frame_files, fps=fps)
    clip.write_videofile(out_filename)

    # Clean up temporary files
    print("Cleaning up temporary files")
    for file in frame_files:
        os.remove(file)
    os.rmdir(tmp_dir)

    print(f"Video with overlaid skeleton saved to {out_filename}")


def main():
    args = parse_args()
    overlay_skeleton_on_video(args.video_file, args.pickle_file, args.out_filename,
                              args.keypoint_threshold, args.fps)


if __name__ == '__main__':
    main()
