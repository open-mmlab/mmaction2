# Demo

## Outline

- [Modify configs through script arguments](#modify-config-through-script-arguments): Tricks to directly modify configs through script arguments.
- [Video demo](#video-demo): A demo script to predict the recognition result using a single video.
- [SpatioTemporal Action Detection Video Demo](#spatiotemporal-action-detection-video-demo): A demo script to predict the SpatioTemporal Action Detection result using a single video.
- [Skeleton-based Action Recognition Demo](#skeleton-based-action-recognition-demo): A demo script to predict the skeleton-based action recognition result using a single video.


## Modify configs through script arguments

When running demos using our provided scripts, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `data.train.pipeline` is normally a list
  e.g. `[dict(type='SampleFrames'), ...]`. If you want to change `'SampleFrames'` to `'DenseSampleFrames'` in the pipeline,
  you may specify `--cfg-options data.train.pipeline.0.type=DenseSampleFrames`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `vis_backends=[dict(type='LocalVisBackend')]`. If you want to
  change this key, you may specify `--cfg-options vis_backends="[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Video demo

We provide a demo script to predict the recognition result using a single video. 

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} {LABEL_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--fps {FPS}] [--font-scale {FONT_SCALE}] [--font-color {FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

Optional arguments:

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it will be set to 30.
- `FONT_SCALE`: Font scale of the label added in the video. If not specified, it will be 0.5.
- `FONT_COLOR`: Font color of the label added in the video. If not specified, it will be `white`.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `RESIZE_ALGORITHM`: Resize algorithm used for resizing. If not specified, it will be set to `bicubic`.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize a video file as input by using a TSN model on cuda by default.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
    ```

2. Recognize a video file as input by using a TSN model on cuda by default, loading checkpoint from url.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
    ```

3. Recognize a list of rawframes as input by using a TSN model on cpu.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --device cpu
    ```

4. Recognize a video file as input by using a TSN model and then generate an mp4 file.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo/demo_out.mp4
    ```

5. Recognize a list of rawframes as input by using a TSN model and then generate a gif file.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --out-filename demo/demo_out.gif
    ```

6. Recognize a video file as input by using a TSN model, then generate an mp4 file with a given resolution and resize algorithm.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --target-resolution 340 256 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    # If either dimension is set to -1, the frames are resized by keeping the existing aspect ratio
    # For --target-resolution 170 -1, original resolution (340, 256) -> target resolution (170, 128)
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --target-resolution 170 -1 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

7. Recognize a video file as input by using a TSN model, then generate an mp4 file with a label in a red color and fontscale 1.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --font-scale 1 --font-color red \
        --out-filename demo/demo_out.mp4
    ```

8. Recognize a list of rawframes as input by using a TSN model and then generate an mp4 file with 24 fps.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --fps 24 --out-filename demo/demo_out.gif
    ```

## SpatioTemporal Action Detection Video Demo

We provide a demo script to predict the SpatioTemporal Action Detection result using a single video.

```shell
python demo/demo_spatiotemporal_det.py --video ${VIDEO_FILE} \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--checkpoint ${SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--output-stepsize ${OUTPUT_STEPSIZE}] \
    [--output-fps ${OUTPUT_FPS}]
```

Optional arguments:

- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The spatiotemporal action detection config file path.
- `SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT`: The spatiotemporal action detection checkpoint URL.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Default: 0.9.
- `ACTION_DETECTION_SCORE_THRESHOLD`: The score threshold for action detection. Default: 0.5.
- `LABEL_MAP`: The label map used. Default: `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Default: `cuda:0`.
- `OUTPUT_FILENAME`: Path to the output file which is a video format. Default: `demo/stdet_demo.mp4`.
- `PREDICT_STEPSIZE`: Make a prediction per N frames.  Default: 8.
- `OUTPUT_STEPSIZE`: Output 1 frame per N frames in the input video. Note that `PREDICT_STEPSIZE % OUTPUT_STEPSIZE == 0`. Default: 4.
- `OUTPUT_FPS`: The FPS of demo video output. Default: 6.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster RCNN as the human detector, SlowOnly-8x8-R101 as the action detector. Making predictions per 8 frames, and output 1 frame per 4 frames to the output video. The FPS of the output video is 4.

```shell
python demo/demo_spatiotemporal_det.py --video demo/demo.mp4 \
    --config configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```

## Skeleton-based Action Recognition Demo

We provide a demo script to predict the skeleton-based action recognition result using a single video.

```shell
python demo/demo_skeleton.py ${VIDEO_FILE} ${OUT_FILENAME} \
    [--config ${SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
    [--checkpoint ${SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--pose-config ${HUMAN_POSE_ESTIMATION_CONFIG_FILE}] \
    [--pose-checkpoint ${HUMAN_POSE_ESTIMATION_CHECKPOINT}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--short-side] ${SHORT_SIDE}
```

Optional arguments:

- `SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE`: The skeleton-based action recognition config file path.
- `SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT`: The skeleton-based action recognition checkpoint path or URL.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Default: 0.9.
- `HUMAN_POSE_ESTIMATION_CONFIG_FILE`: The human pose estimation config file path (trained on COCO-Keypoint).
- `HUMAN_POSE_ESTIMATION_CHECKPOINT`: The human pose estimation checkpoint URL (trained on COCO-Keypoint).
- `LABEL_MAP`: The label map used. Default: `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Default: `cuda:0`.
- `SHORT_SIDE`: The short side used for frame extraction. Default: 480.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster RCNN as the human detector, HRNetw32 as the pose estimator, PoseC3D-NTURGB+D-120-Xsub-keypoint as the skeleton-based action recognizer.

```shell
python demo/demo_skeleton.py demo/ntu_sample.avi demo/skeleton_demo.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --pose-config demo/hrnet_w32_coco_256x192.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu120.txt
```

2. Use the Faster RCNN as the human detector, HRNetw32 as the pose estimator, STGCN-NTURGB+D-60-Xsub-keypoint as the skeleton-based action recognizer.

```shell
python demo/demo_skeleton.py demo/ntu_sample.avi demo/skeleton_demo.mp4 \
    --config configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --pose-config demo/hrnet_w32_coco_256x192.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu120.txt
```


Demo script to predict the audio-based action recognition using a single audio feature.

The script `extract_audio.py` can be used to extract audios from videos and the script `build_audio_features.py` can be used to extract the audio features.

```shell
python demo/demo_audio.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${AUDIO_FILE} {LABEL_FILE} [--device ${DEVICE}]
```

Optional arguments:

- `DEVICE`: Type of device to run the demo. Allowed values are cuda devices like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load the corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize an audio file as input by using a tsn model on cuda by default.

    ```shell
    python demo/demo_audio.py \
        configs/recognition_audio/resnet/tsn_r18_64x1x1_100e_kinetics400_audio_feature.py \
        https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth \
        audio_feature.npy label_map_k400.txt
    ```
