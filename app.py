"""
FastAPI application for skeleton-based action recognition.
"""
from typing import Optional, Dict, Any
from pathlib import Path
import os
import uuid
import shutil
import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

# Import the processing functions from demo_skeleton_refactored
from demo.demo_skeleton_refactored import (
    process_video_windows,
    visualize_with_labels,
    frame_extract,
    detection_inference,
    pose_inference,
    init_recognizer
)


class ProcessingConfig:
    """Configuration for video processing."""

    def __init__(self):
        # Create timestamped root directory for better organization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root_dir = Path(f"skeleton_recognition_{timestamp}")
        self.upload_dir = self.root_dir / "uploads"
        self.output_dir = self.root_dir / "processed_videos"
        self.logs_dir = self.root_dir / "logs"

        # Model configurations based on infer_skl.sh
        self.model_configs = {
            "config": "work_dirs/posec3d_ntu60_2d_adam/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py",
            "checkpoint": "work_dirs/posec3d_ntu60_2d_adam/best_acc_top1_epoch_24.pth",
            "det_config": "demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py",
            "det_checkpoint": "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth",
            "pose_config": "demo/demo_configs/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
            "pose_checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth",
            "label_map": "tools/data/skeleton/label_map_ntu60.txt"
        }

        # Processing parameters
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.short_side = 480
        self.window_size = 32
        self.window_stride = 16
        self.det_score_thr = 0.9
        self.det_cat_id = 0

        # Create necessary directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create all necessary directories for processing."""
        self.root_dir.mkdir(exist_ok=True)
        self.upload_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)


class VideoResponse(BaseModel):
    """Response model for video processing."""
    video_id: str
    processed_video_path: str
    full_url: str
    message: str
    processing_info: Dict[str, Any]


app = FastAPI(
    title="Skeleton-based Action Recognition API",
    description="API for processing videos using skeleton-based action recognition",
    version="1.0.0"
)

config = ProcessingConfig()

# Mount the processed videos directory for direct access
app.mount("/videos", StaticFiles(directory=str(config.output_dir)), name="videos")


@app.post("/process_video/", response_model=VideoResponse)
async def process_video(
    video: UploadFile = File(...),
    det_score_thr: float = Query(0.9, description="Detection score threshold")
) -> VideoResponse:
    """
    Process a video file using skeleton-based action recognition.

    Args:
        video (UploadFile): The input video file to process
        det_score_thr (float): Detection score threshold

    Returns:
        VideoResponse: Object containing the processed video information

    Raises:
        HTTPException: If video processing fails
    """
    try:
        # Generate unique filename for the upload
        video_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create paths for input and output files
        video_filename = Path(video.filename).stem
        video_extension = Path(video.filename).suffix
        input_path = config.upload_dir / \
            f"{video_id}_{video_filename}{video_extension}"
        output_filename = f"{video_id}_{video_filename}_processed.mp4"
        output_path = config.output_dir / output_filename

        # Log processing request
        log_path = config.logs_dir / f"{video_id}_processing.log"
        with open(log_path, "w") as log_file:
            log_file.write(f"Processing started at: {timestamp}\n")
            log_file.write(f"Input video: {video.filename}\n")
            log_file.write(f"Detection score threshold: {det_score_thr}\n")

        # Save uploaded video
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Update config with request-specific parameters
        config.det_score_thr = det_score_thr

        # Extract frames
        frame_paths, frames = frame_extract(
            str(input_path),
            config.short_side,
            str(config.upload_dir)
        )
        h, w, _ = frames[0].shape

        # Perform detection
        det_results, _ = detection_inference(
            config.model_configs["det_config"],
            config.model_configs["det_checkpoint"],
            frame_paths,
            config.det_score_thr,
            config.det_cat_id,
            config.device
        )
        torch.cuda.empty_cache()

        # Perform pose estimation
        pose_results, pose_data_samples = pose_inference(
            config.model_configs["pose_config"],
            config.model_configs["pose_checkpoint"],
            frame_paths,
            det_results,
            config.device
        )
        torch.cuda.empty_cache()

        # Initialize model
        model = init_recognizer(
            config.model_configs["config"],
            config.model_configs["checkpoint"],
            config.device
        )

        # Load labels
        with open(config.model_configs["label_map"], 'r') as f:
            label_map = [x.strip() for x in f.readlines()]

        # Process video windows
        frame_labels, frame_confidences = process_video_windows(
            model,
            pose_results,
            (h, w),
            label_map,
            config.window_size,
            config.window_stride,
            len(pose_results)
        )

        # Create visualization
        args = type('Args', (), {
            'out_filename': str(output_path),
            'video': str(input_path),
            'det_score_thr': config.det_score_thr,
            'pose_config': config.model_configs["pose_config"],
            'pose_checkpoint': config.model_configs["pose_checkpoint"],
            'config': config.model_configs["config"],
            'checkpoint': config.model_configs["checkpoint"],
            'label_map': config.model_configs["label_map"],
            'device': config.device,
            'short_side': config.short_side,
            'window_size': config.window_size,
            'window_stride': config.window_stride,
            'det_cat_id': config.det_cat_id,
            'cfg_options': {}
        })

        visualize_with_labels(
            args,
            frames,
            pose_data_samples,
            frame_labels,
            frame_confidences
        )

        # Generate full URL for accessing the video
        # Change this to your actual host URL in production
        host_url = "http://localhost:8000"
        video_url = f"{host_url}/videos/{output_filename}"

        # Prepare processing info
        processing_info = {
            "timestamp": timestamp,
            "input_video": str(input_path),
            "output_video": str(output_path),
            "detection_score_threshold": config.det_score_thr,
            "num_frames_processed": len(frames),
            "device_used": config.device
        }

        # Log completion
        with open(log_path, "a") as log_file:
            log_file.write(
                f"Processing completed at: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            log_file.write(f"Output video: {str(output_path)}\n")

        return VideoResponse(
            video_id=video_id,
            processed_video_path=str(output_path),
            full_url=video_url,
            message="Video processed successfully",
            processing_info=processing_info
        )

    except Exception as e:
        # Log error
        error_log_path = config.logs_dir / \
            f"{video_id}_error.log" if 'video_id' in locals(
            ) else config.logs_dir / f"error_{timestamp}.log"
        with open(error_log_path, "w") as error_file:
            error_file.write(
                f"Error occurred at: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
            error_file.write(f"Error details: {str(e)}\n")

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/video/{video_id}")
async def get_video(video_id: str):
    """
    Retrieve a processed video file.

    Args:
        video_id (str): The ID of the processed video

    Returns:
        FileResponse: The video file

    Raises:
        HTTPException: If video file is not found
    """
    # Find the video with the given ID (with any filename)
    for file in config.output_dir.glob(f"{video_id}_*"):
        if file.is_file() and file.suffix in (".mp4", ".avi", ".mov"):
            return FileResponse(
                str(file),
                media_type="video/mp4",
                filename=file.name
            )

    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/videos/")
async def list_videos():
    """
    List all processed videos.

    Returns:
        JSONResponse: List of available videos with their IDs and paths
    """
    videos = []
    for file in config.output_dir.glob("*"):
        if file.is_file() and file.suffix in (".mp4", ".avi", ".mov"):
            video_id = file.stem.split("_")[0]
            videos.append({
                "video_id": video_id,
                "filename": file.name,
                "path": str(file),
                "url": f"/videos/{file.name}"
            })

    return JSONResponse(content={"videos": videos})


@app.on_event("startup")
async def startup_event():
    """Initialize directories and models on startup."""
    config._create_directories()
    print(f"Server initialized with the following configuration:")
    print(f"- Root directory: {config.root_dir}")
    print(f"- Upload directory: {config.upload_dir}")
    print(f"- Output directory: {config.output_dir}")
    print(f"- Logs directory: {config.logs_dir}")
    print(f"- Using device: {config.device}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
