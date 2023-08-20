# Demo

## Outline

- [Modify configs through script arguments](#modify-config-through-script-arguments): Tricks to directly modify configs through script arguments.
- [Video demo](#video-demo): A demo script to predict the recognition result using a single video.
- [Video GradCAM Demo](#video-gradcam-demo): A demo script to visualize GradCAM results using a single video.
- [Webcam demo](#webcam-demo): A demo script to implement real-time action recognition from a web camera.
- [Long Video demo](#long-video-demo): a demo script to predict different labels using a single long video.
- [Skeleton-based Action Recognition Demo](#skeleton-based-action-recognition-demo): A demo script to predict the skeleton-based action recognition result using a single video.
- [SpatioTemporal Action Detection Webcam Demo](#spatiotemporal-action-detection-webcam-demo): A demo script to implement real-time spatio-temporal action detection from a web camera.
- [SpatioTemporal Action Detection Video Demo](#spatiotemporal-action-detection-video-demo): A demo script to predict the spatiotemporal action detection result using a single video.
- [SpatioTemporal Action Detection ONNX Video Demo](#spatiotemporal-action-detection-onnx-video-demo): A demo script to predict the SpatioTemporal Action Detection result using the onnx file instead of building the PyTorch models.
- [Inferencer Demo](#inferencer): A demo script to implement fast predict for video analysis tasks based on unified inferencer interface.
- [Audio Demo](#audio-demo): A demo script to predict the recognition result using a single audio file.
- [Video Structuralize Demo](#video-structuralize-demo): A demo script to predict the skeleton-based and rgb-based action recognition and spatio-temporal action detection result using a single video.

## Modify configs through script arguments

When running demos using our provided scripts, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_dataloader.dataset.pipeline` is normally a list
  e.g. `[dict(type='SampleFrames'), ...]`. If you want to change `'SampleFrames'` to `'DenseSampleFrames'` in the pipeline,
  you may specify `--cfg-options train_dataloader.dataset.pipeline.0.type=DenseSampleFrames`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark " is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Video demo

MMAction2 provides a demo script to predict the recognition result using a single video. In order to get predict results in range `[0, 1]`, make sure to set `model['test_cfg'] = dict(average_clips='prob')` in config file.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--fps ${FPS}] [--font-scale ${FONT_SCALE}] [--font-color ${FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--out-filename ${OUT_FILE}]
```

Optional arguments:

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it will be set to 30.
- `FONT_SCALE`: Font scale of the text added in the video. If not specified, it will be None.
- `FONT_COLOR`: Font color of the text added in the video. If not specified, it will be `white`.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize a video file as input by using a TSN model on cuda by default.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```

2. Recognize a video file as input by using a TSN model on cuda by default, loading checkpoint from url.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
       https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```

3. Recognize a video file as input by using a TSN model and then generate an mp4 file.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo/demo_out.mp4
   ```

## Video GradCAM Demo

MMAction2 provides a demo script to visualize GradCAM results using a single video.

```shell
python tools/visualizations/vis_cam.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--target-layer-name ${TARGET_LAYER_NAME}] [--fps {FPS}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it will be set to 30.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.
- `TARGET_LAYER_NAME`: Layer name to generate GradCAM localization map.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `RESIZE_ALGORITHM`: Resize algorithm used for resizing. If not specified, it will be set to `bilinear`.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Get GradCAM results of a I3D model, using a video file as input and then generate an gif file with 10 fps.

   ```shell
   python tools/visualizations/vis_cam.py demo/demo_configs/i3d_r50_32x2x1_video_infer.py \
       checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth demo/demo.mp4 \
       --target-layer-name backbone/layer4/1/relu --fps 10 \
       --out-filename demo/demo_gradcam.gif
   ```

2. Get GradCAM results of a TSN model, using a video file as input and then generate an gif file, loading checkpoint from url.

   ```shell
   python tools/visualizations/vis_cam.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
       https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb_20220906-dcbc6e01.pth \
       demo/demo.mp4 --target-layer-name backbone/layer4/1/relu --out-filename demo/demo_gradcam_tsn.gif
   ```

## Webcam demo

We provide a demo script to implement real-time action recognition from web camera. In order to get predict results in range `[0, 1]`, make sure to set `model.cls_head.average_clips='prob'` in config file.

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--camera-id ${CAMERA_ID}] [--threshold ${THRESHOLD}] \
    [--average-size ${AVERAGE_SIZE}] [--drawing-fps ${DRAWING_FPS}] [--inference-fps ${INFERENCE_FPS}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `CAMERA_ID`: ID of camera device If not specified, it will be set to 0.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.
- `AVERAGE_SIZE`: Number of latest clips to be averaged for prediction. If not specified, it will be set to 1.
- `DRAWING_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 20.
- `INFERENCE_FPS`: Upper bound FPS value of the output drawing. If not specified, it will be set to 4.

If your hardware is good enough, increasing the value of `DRAWING_FPS` and `INFERENCE_FPS` will get a better experience.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
   and outputting result labels with score higher than 0.2.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth tools/data/kinetics/label_map_k400.txt --average-size 5 \
     --threshold 0.2 --device cpu
   ```

2. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
     tools/data/kinetics/label_map_k400.txt --average-size 5 --threshold 0.2 --device cpu
   ```

3. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
   and outputting result labels with score higher than 0.2.

   ```shell
   python demo/webcam_demo.py demo/demo_configs/i3d_r50_32x2x1_video_infer.py \
     checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth tools/data/kinetics/label_map_k400.txt \
     --average-size 5 --threshold 0.2
   ```

Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case.
Users can change:

- `SampleFrames` step (especially the number of `clip_len` and `num_clips`) of `test_pipeline` in the config file, like `--cfg-options test_pipeline.0.num_clips=3`.
- Change to the suitable Crop methods like `TenCrop`, `ThreeCrop`, `CenterCrop`, etc. in `test_pipeline` of the config file, like `--cfg-options test_pipeline.4.type=CenterCrop`.
- Change the number of `--average-size`. The smaller, the faster.

## Long video demo

We provide a demo script to predict different labels using a single long video. In order to get predict results in range `[0, 1]`, make sure to set `cls_head = dict(average_clips='prob')` in config file.

```shell
python demo/long_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    ${OUT_FILE} [--input-step ${INPUT_STEP}] [--device ${DEVICE_TYPE}] [--threshold ${THRESHOLD}]
```

Optional arguments:

- `OUT_FILE`: Path to the output, either video or json file
- `INPUT_STEP`: Input step for sampling frames, which can help to get more spare input. If not specified , it will be set to 1.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.01.
- `STRIDE`: By default, the demo generates a prediction for each single frame, which might cost lots of time. To speed up, you can set the argument `STRIDE` and then the demo will generate a prediction every `STRIDE x sample_length` frames (`sample_length` indicates the size of temporal window from which you sample frames, which equals to `clip_len x frame_interval`). For example, if the sample_length is 64 frames and you set `STRIDE` to 0.5, predictions will be generated every 32 frames. If set as 0, predictions will be generated for each frame. The desired value of `STRIDE` is (0, 1\], while it also works for `STRIDE > 1` (the generated predictions will be too sparse). Default: 0.
- `LABEL_COLOR`: Font Color of the labels in (B, G, R). Default is white, that is (256, 256, 256).
- `MSG_COLOR`: Font Color of the messages in (B, G, R). Default is gray, that is (128, 128, 128).

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Predict different labels in a long video by using a TSN model on cpu, with 8 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2.

   ```shell
   python demo/long_video_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth PATH_TO_LONG_VIDEO tools/data/kinetics/label_map_k400.txt PATH_TO_SAVED_VIDEO \
     --input-step 3 --device cpu --threshold 0.2
   ```

2. Predict different labels in a long video by using a TSN model on cpu, with 8 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

   ```shell
   python demo/long_video_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
     PATH_TO_LONG_VIDEO tools/data/kinetics/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
   ```

3. Predict different labels in a long video from web by using a TSN model on cpu, with 8 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

   ```shell
   python demo/long_video_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
     https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
     https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4 \
     tools/data/kinetics/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
   ```

4. Predict different labels in a long video by using a I3D model on gpu, with input_step=1, threshold=0.01 as default and print the labels in cyan.

   ```shell
   python demo/long_video_demo.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
     checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth PATH_TO_LONG_VIDEO tools/data/kinetics/label_map_k400.txt PATH_TO_SAVED_VIDEO \
     --label-color 255 255 0
   ```

5. Predict different labels in a long video by using a I3D model on gpu and save the results as a `json` file

   ```shell
   python demo/long_video_demo.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
     checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth PATH_TO_LONG_VIDEO tools/data/kinetics/label_map_k400.txt ./results.json
   ```

## Skeleton-based Action Recognition Demo

MMAction2 provides a demo script to predict the skeleton-based action recognition result using a single video.

```shell
python demo/demo_skeleton.py ${VIDEO_FILE} ${OUT_FILENAME} \
    [--config ${SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
    [--checkpoint ${SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--det-cat-id ${HUMAN_DETECTION_CATEGORY_ID}] \
    [--pose-config ${HUMAN_POSE_ESTIMATION_CONFIG_FILE}] \
    [--pose-checkpoint ${HUMAN_POSE_ESTIMATION_CHECKPOINT}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--short-side] ${SHORT_SIDE}
```

Optional arguments:

- `SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE`: The skeleton-based action recognition config file path.
- `SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT`: The skeleton-based action recognition checkpoint path or url.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint path or url.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Defaults to 0.9.
- `HUMAN_DETECTION_CATEGORY_ID`: The category id for human detection. Defaults to 0.
- `HUMAN_POSE_ESTIMATION_CONFIG_FILE`: The human pose estimation config file path (trained on COCO-Keypoint).
- `HUMAN_POSE_ESTIMATION_CHECKPOINT`: The human pose estimation checkpoint path or url (trained on COCO-Keypoint).
- `LABEL_MAP`: The label map used. Defaults to `'tools/data/skeleton/label_map_ntu60.txt'`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `'cuda:0'` or `'cpu'`. Defaults to `'cuda:0'`.
- `SHORT_SIDE`: The short side used for frame extraction. Defaults to 480.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster-RCNN as the human detector, HRNetw32 as the pose estimator, PoseC3D-NTURGB+D-60-XSub-Keypoint as the skeleton-based action recognizer.

```shell
python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```

2. Use the Faster-RCNN as the human detector, HRNetw32 as the pose estimator, STGCN-NTURGB+D-60-XSub-Keypoint as the skeleton-based action recognizer.

```shell
python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
    --config configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --checkpoint https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```

## SpatioTemporal Action Detection Webcam Demo

We provide a demo script to implement real-time spatio-temporal action detection from a web camera.

```shell
python demo/webcam_demo_spatiotemporal_det.py \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--checkpoint ${SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--input-video] ${INPUT_VIDEO} \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--output-fps ${OUTPUT_FPS}] \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--show] \
    [--display-height] ${DISPLAY_HEIGHT} \
    [--display-width] ${DISPLAY_WIDTH} \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--clip-vis-length] ${CLIP_VIS_LENGTH}
```

Optional arguments:

- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The spatiotemporal action detection config file path.
- `SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT`: The spatiotemporal action detection checkpoint path or URL.
- `ACTION_DETECTION_SCORE_THRESHOLD`: The score threshold for action detection. Default: 0.4.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Default: 0.9.
- `INPUT_VIDEO`: The webcam id or video path of the source. Default: `0`.
- `LABEL_MAP`: The label map used. Default: `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Default: `cuda:0`.
- `OUTPUT_FPS`: The FPS of demo video output. Default: 15.
- `OUTPUT_FILENAME`: Path to the output file which is a video format. Default: None.
- `--show`: Whether to show predictions with `cv2.imshow`.
- `DISPLAY_HEIGHT`: The height of the display frame. Default: 0.
- `DISPLAY_WIDTH`: The width of the display frame. Default: 0. If `DISPLAY_HEIGHT <= 0 and DISPLAY_WIDTH <= 0`, the display frame and input video share the same shape.
- `PREDICT_STEPSIZE`: Make a prediction per N frames. Default: 8.
- `CLIP_VIS_LENGTH`: The number of the draw frames for each clip. In other words, for each clip, there are at most `CLIP_VIS_LENGTH` frames to be draw around the keyframe. DEFAULT: 8.

Tips to get a better experience for webcam demo:

- How to choose `--output-fps`?

  - `--output-fps` should be almost equal to read thread fps.
  - Read thread fps is printed by logger in format `DEBUG:__main__:Read Thread: {duration} ms, {fps} fps`

- How to choose `--predict-stepsize`?

  - It's related to how to choose human detector and spatio-temporval model.
  - Overall, the duration of read thread for each task should be greater equal to that of model inference.
  - The durations for read/inference are both printed by logger.
  - Larger `--predict-stepsize` leads to larger duration for read thread.
  - In order to fully take the advantage of computation resources, decrease the value of `--predict-stepsize`.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster RCNN as the human detector, SlowOnly-8x8-R101 as the action detector. Making predictions per 40 frames, and FPS of the output is 20. Show predictions with `cv2.imshow`.

```shell
python demo/webcam_demo_spatiotemporal_det.py \
    --input-video 0 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 40 \
    --output-fps 20 \
    --show
```

## SpatioTemporal Action Detection Video Demo

MMAction2 provides a demo script to predict the SpatioTemporal Action Detection result using a single video.

```shell
python demo/demo_spatiotemporal_det.py --video ${VIDEO_FILE} \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--checkpoint ${SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--det-cat-id ${HUMAN_DETECTION_CATEGORY_ID}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--short-side] ${SHORT_SIDE} \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--output-stepsize ${OUTPUT_STEPSIZE}] \
    [--output-fps ${OUTPUT_FPS}]
```

Optional arguments:

- `OUTPUT_FILENAME`: Path to the output file which is a video format. Defaults to `demo/stdet_demo.mp4`.
- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The spatiotemporal action detection config file path.
- `SPATIOTEMPORAL_ACTION_DETECTION_CHECKPOINT`: The spatiotemporal action detection checkpoint URL.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRESHOLD`: The score threshold for human detection. Defaults to 0.9.
- `HUMAN_DETECTION_CATEGORY_ID`: The category id for human detection. Defaults to 0.
- `ACTION_DETECTION_SCORE_THRESHOLD`: The score threshold for action detection. Defaults to 0.5.
- `LABEL_MAP`: The label map used. Defaults to `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Defaults to `cuda:0`.
- `SHORT_SIDE`: The short side used for frame extraction. Defaults to 256.
- `PREDICT_STEPSIZE`: Make a prediction per N frames.  Defaults to 8.
- `OUTPUT_STEPSIZE`: Output 1 frame per N frames in the input video. Note that `PREDICT_STEPSIZE % OUTPUT_STEPSIZE == 0`. Defaults to 4.
- `OUTPUT_FPS`: The FPS of demo video output. Defaults to 6.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster RCNN as the human detector, SlowOnly-8x8-R101 as the action detector. Making predictions per 8 frames, and output 1 frame per 4 frames to the output video. The FPS of the output video is 4.

```shell
python demo/demo_spatiotemporal_det.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```

## SpatioTemporal Action Detection ONNX Video Demo

MMAction2 provides a demo script to predict the SpatioTemporal Action Detection result using the onnx file instead of building the PyTorch models.

```shell
python demo/demo_spatiotemporal_det_onnx.py --video ${VIDEO_FILE} \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--config ${SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--onnx-file ${SPATIOTEMPORAL_ACTION_DETECTION_ONNX_FILE}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRESHOLD}] \
    [--det-cat-id ${HUMAN_DETECTION_CATEGORY_ID}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRESHOLD}] \
    [--label-map ${LABEL_MAP}] \
    [--device ${DEVICE}] \
    [--short-side] ${SHORT_SIDE} \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--output-stepsize ${OUTPUT_STEPSIZE}] \
    [--output-fps ${OUTPUT_FPS}]
```

Optional arguments:

- `OUTPUT_FILENAME`: Path to the output file which is a video format. Defaults to `demo/stdet_demo.mp4`.
- `SPATIOTEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The spatiotemporal action detection config file path.
- `SPATIOTEMPORAL_ACTION_DETECTION_ONNX_FILE`: The spatiotemporal action detection onnx file.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_DETECTION_SCORE_THRESHOLD`: The score threshold for human detection. Defaults to 0.9.
- `HUMAN_DETECTION_CATEGORY_ID`: The category id for human detection. Defaults to 0.
- `ACTION_DETECTION_SCORE_THRESHOLD`: The score threshold for action detection. Defaults to 0.5.
- `LABEL_MAP`: The label map used. Defaults to `tools/data/ava/label_map.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Defaults to `cuda:0`.
- `SHORT_SIDE`: The short side used for frame extraction. Defaults to 256.
- `PREDICT_STEPSIZE`: Make a prediction per N frames.  Defaults to 8.
- `OUTPUT_STEPSIZE`: Output 1 frame per N frames in the input video. Note that `PREDICT_STEPSIZE % OUTPUT_STEPSIZE == 0`. Defaults to 4.
- `OUTPUT_FPS`: The FPS of demo video output. Defaults to 6.

Examples:

Assume that you are located at `$MMACTION2` .

1. Export an onnx file given the config file and checkpoint.

```shell
python tools/deployment/export_onnx_stdet.py \
    configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --output_file slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx \
    --num_frames 8
```

2. Use the Faster RCNN as the human detector, the generated `slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx` file as the action detector. Making predictions per 8 frames, and output 1 frame per 4 frames to the output video. The FPS of the output video is 4.

```shell
python demo/demo_spatiotemporal_det_onnx.py demo/demo.mp4 demo/demo_spatiotemporal_det.mp4 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --onnx-file slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```

## Inferencer

MMAction2 provides a demo script to implement fast prediction for video analysis tasks based on unified inferencer interface, currently only supports action recognition task.

```shell
python demo/demo.py ${INPUTS} \
    [--vid-out-dir ${VID_OUT_DIR}] \
    [--rec ${RECOG_TASK}] \
    [--rec-weights ${RECOG_WEIGHTS}] \
    [--label-file ${LABEL_FILE}] \
    [--device ${DEVICE_TYPE}] \
    [--batch-size ${BATCH_SIZE}] \
    [--print-result ${PRINT_RESULT}] \
    [--pred-out-file ${PRED_OUT_FILE} ]
```

Optional arguments:

- `--show`: If specified, the demo will display the video in a popup window.
- `--print-result`: If specified, the demo will print the inference results'
- `VID_OUT_DIR`: Output directory of saved videos. Defaults to None, means not to save videos.
- `RECOG_TASK`: Type of Action Recognition algorithm. It could be the path to the config file, the model name or alias defined in metafile.
- `RECOG_WEIGHTS`: Path to the custom checkpoint file of the selected recog model. If it is not specified and "rec" is a model name of metafile, the weights will be loaded from metafile.
- `LABEL_FILE`: Label file for dataset the algorithm pretrained on. Defaults to None, means don't show label in result.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. Defaults to `cuda:0`.
- `BATCH_SIZE`: The batch size used in inference. Defaults to 1.
- `PRED_OUT_FILE`: File path to save the inference results. Defaults to None, means not to save prediction results.

Examples:

Assume that you are located at `$MMACTION2`.

1. Recognize a video file as input by using a TSN model, loading checkpoint from metafile.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo_inferencer.py demo/demo.mp4 \
       --rec tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb \
       --label-file tools/data/kinetics/label_map_k400.txt
   ```

2. Recognize a video file as input by using a TSN model, using model alias in metafile.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo_inferencer.py demo/demo.mp4 \
       --rec tsn \
       --label-file tools/data/kinetics/label_map_k400.txt
   ```

3. Recognize a video file as input by using a TSN model, and then save visulization video.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo_inferencer.py demo/demo.mp4 \
       --vid-out-dir demo_out \
       --rec tsn \
       --label-file tools/data/kinetics/label_map_k400.txt
   ```

## Audio Demo

Demo script to predict the audio-based action recognition using a single audio feature.

The script [`extract_audio.py`](/tools/data/extract_audio.py) can be used to extract audios from videos and the script [`build_audio_features.py`](/tools/data/build_audio_features.py) can be used to extract the audio features.

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
       configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
       https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature_20230702-e4642fb0.pth \
       audio_feature.npy tools/data/kinetics/label_map_k400.txt
   ```

## Video Structuralize Demo

We provide a demo script to predict the skeleton-based and rgb-based action recognition and spatio-temporal action detection result using a single video.

```shell
python demo/demo_video_structuralize.py \
    [--rgb-stdet-config ${RGB_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CONFIG_FILE}] \
    [--rgb-stdet-checkpoint ${RGB_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--skeleton-stdet-checkpoint ${SKELETON_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CHECKPOINT}] \
    [--det-config ${HUMAN_DETECTION_CONFIG_FILE}] \
    [--det-checkpoint ${HUMAN_DETECTION_CHECKPOINT}] \
    [--pose-config ${HUMAN_POSE_ESTIMATION_CONFIG_FILE}] \
    [--pose-checkpoint ${HUMAN_POSE_ESTIMATION_CHECKPOINT}] \
    [--skeleton-config ${SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
    [--skeleton-checkpoint ${SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
    [--rgb-config ${RGB_BASED_ACTION_RECOGNITION_CONFIG_FILE}] \
    [--rgb-checkpoint ${RGB_BASED_ACTION_RECOGNITION_CHECKPOINT}] \
    [--use-skeleton-stdet ${USE_SKELETON_BASED_SPATIO_TEMPORAL_DETECTION_METHOD}] \
    [--use-skeleton-recog ${USE_SKELETON_BASED_ACTION_RECOGNITION_METHOD}] \
    [--det-score-thr ${HUMAN_DETECTION_SCORE_THRE}] \
    [--action-score-thr ${ACTION_DETECTION_SCORE_THRE}] \
    [--video ${VIDEO_FILE}] \
    [--label-map-stdet ${LABEL_MAP_FOR_SPATIO_TEMPORAL_ACTION_DETECTION}] \
    [--device ${DEVICE}] \
    [--out-filename ${OUTPUT_FILENAME}] \
    [--predict-stepsize ${PREDICT_STEPSIZE}] \
    [--output-stepsize ${OUTPU_STEPSIZE}] \
    [--output-fps ${OUTPUT_FPS}] \
    [--cfg-options]
```

Optional arguments:

- `RGB_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CONFIG_FILE`: The rgb-based spatio temoral action detection config file path.
- `RGB_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CHECKPOINT`: The rgb-based spatio temoral action detection checkpoint path or URL.
- `SKELETON_BASED_SPATIO_TEMPORAL_ACTION_DETECTION_CHECKPOINT`: The skeleton-based spatio temoral action detection checkpoint path or URL.
- `HUMAN_DETECTION_CONFIG_FILE`: The human detection config file path.
- `HUMAN_DETECTION_CHECKPOINT`: The human detection checkpoint URL.
- `HUMAN_POSE_ESTIMATION_CONFIG_FILE`: The human pose estimation config file path (trained on COCO-Keypoint).
- `HUMAN_POSE_ESTIMATION_CHECKPOINT`: The human pose estimation checkpoint URL (trained on COCO-Keypoint).
- `SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE`: The skeleton-based action recognition config file path.
- `SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT`: The skeleton-based action recognition checkpoint path or URL.
- `RGB_BASED_ACTION_RECOGNITION_CONFIG_FILE`: The rgb-based action recognition config file path.
- `RGB_BASED_ACTION_RECOGNITION_CHECKPOINT`: The rgb-based action recognition checkpoint path or URL.
- `USE_SKELETON_BASED_SPATIO_TEMPORAL_DETECTION_METHOD`: Use skeleton-based spatio temporal action detection method.
- `USE_SKELETON_BASED_ACTION_RECOGNITION_METHOD`: Use skeleton-based action recognition method.
- `HUMAN_DETECTION_SCORE_THRE`: The score threshold for human detection. Default: 0.9.
- `ACTION_DETECTION_SCORE_THRE`: The score threshold for action detection. Default: 0.4.
- `LABEL_MAP_FOR_SPATIO_TEMPORAL_ACTION_DETECTION`: The label map for spatio temporal action detection used. Default: `tools/data/ava/label_map.txt`.
- `LABEL_MAP`: The label map for action recognition. Default: `tools/data/kinetics/label_map_k400.txt`.
- `DEVICE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`.  Default: `cuda:0`.
- `OUTPUT_FILENAME`: Path to the output file which is a video format. Default: `demo/test_stdet_recognition_output.mp4`.
- `PREDICT_STEPSIZE`: Make a prediction per N frames.  Default: 8.
- `OUTPUT_STEPSIZE`: Output 1 frame per N frames in the input video. Note that `PREDICT_STEPSIZE % OUTPUT_STEPSIZE == 0`. Default: 1.
- `OUTPUT_FPS`: The FPS of demo video output. Default: 24.

Examples:

Assume that you are located at `$MMACTION2` .

1. Use the Faster RCNN as the human detector, HRNetw32 as the pose estimator, PoseC3D as the skeleton-based action recognizer and the skeleton-based spatio temporal action detector. Making action detection predictions per 8 frames, and output 1 frame per 1 frame to the output video. The FPS of the output video is 24.

```shell
python demo/demo_video_structuralize.py \
    --skeleton-stdet-checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_ava.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --skeleton-checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_k400.pth \
    --use-skeleton-stdet \
    --use-skeleton-recog \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt
```

2. Use the Faster RCNN as the human detector, TSN-R50-1x1x3 as the rgb-based action recognizer, SlowOnly-8x8-R101 as the rgb-based spatio temporal action detector. Making action detection predictions per 8 frames, and output 1 frame per 1 frame to the output video. The FPS of the output video is 24.

```shell
python demo/demo_video_structuralize.py \
    --rgb-stdet-config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --rgb-stdet-checkpoint  https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --rgb-config demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
    --rgb-checkpoint https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt
```

3. Use the Faster RCNN as the human detector, HRNetw32 as the pose estimator, PoseC3D as the skeleton-based action recognizer, SlowOnly-8x8-R101 as the rgb-based spatio temporal action detector. Making action detection predictions per 8 frames, and output 1 frame per 1 frame to the output video. The FPS of the output video is 24.

```shell
python demo/demo_video_structuralize.py \
    --rgb-stdet-config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --rgb-stdet-checkpoint  https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --skeleton-checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_k400.pth \
    --use-skeleton-recog \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt
```

4. Use the Faster RCNN as the human detector, HRNetw32 as the pose estimator, TSN-R50-1x1x3 as the rgb-based action recognizer, PoseC3D as the skeleton-based spatio temporal action detector. Making action detection predictions per 8 frames, and output 1 frame per 1 frame to the output video. The FPS of the output video is 24.

```shell
python demo/demo_video_structuralize.py
    --skeleton-stdet-checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/posec3d_ava.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --rgb-config demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
    --rgb-checkpoint https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
    --use-skeleton-stdet \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt
```
