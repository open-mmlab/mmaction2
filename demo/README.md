# Demo

## Outline

- [Modify configs through script arguments](#modify-config-through-script-arguments): Tricks to directly modify configs through script arguments.
- [Video demo](#video-demo): A demo script to predict the recognition result using a single video.
- [SpatioTemporal Action Detection Video Demo](#spatiotemporal-action-detection-video-demo): A demo script to predict the SpatioTemporal Action Detection result using a single video.
- [Video GradCAM Demo](#video-gradcam-demo): A demo script to visualize GradCAM results using a single video.
- [Webcam demo](#webcam-demo): A demo script to implement real-time action recognition from a web camera.
- [Long Video demo](#long-video-demo): a demo script to predict different labels using a single long video.
- [SpatioTempoval Action Detection Webcam Demo](#spatiotemporal-action-detection-webcam-demo): A demo script to implement real-time spatio-temporval action detection from a web camera.

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

  If the value to be updated is a list or a tuple. For example, the config file normally sets `workflow=[('train', 1)]`. If you want to
  change this key, you may specify `--cfg-options workflow="[(train,1),(val,1)]"`. Note that the quotation mark \" is necessary to
  support list/tuple data types, and that **NO** white space is allowed inside the quotation marks in the specified value.

## Video demo

We provide a demo script to predict the recognition result using a single video. In order to get predict results in range `[0, 1]`, make sure to set `model['test_cfg'] = dict(average_clips='prob')` in config file.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} {LABEL_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--fps {FPS}] [--font-scale {FONT_SCALE}] [--font-color {FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

Optional arguments:

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it wll be set to 30.
- `FONT_SCALE`: Font scale of the label added in the video. If not specified, it wll be 0.5.
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
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt
    ```

2. Recognize a video file as input by using a TSN model on cuda by default, loading checkpoint from url.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt
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
        demo/demo.mp4 demo/label_map_k400.txt --out-filename demo/demo_out.mp4
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
        demo/demo.mp4 demo/label_map_k400.txt --target-resolution 340 256 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    # If either dimension is set to -1, the frames are resized by keeping the existing aspect ratio
    # For --target-resolution 170 -1, original resolution (340, 256) -> target resolution (170, 128)
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --target-resolution 170 -1 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

7. Recognize a video file as input by using a TSN model, then generate an mp4 file with a label in a red color and fontscale 1.

    ```shell
    # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map_k400.txt --font-scale 1 --font-color red \
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
- `LABEL_MAP`: The label map used. Default: `demo/label_map_ava.txt`.
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
    --label-map demo/label_map_ava.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
```

## Video GradCAM Demo

We provide a demo script to visualize GradCAM results using a single video.

```shell
python demo/demo_gradcam.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--target-layer-name ${TARGET_LAYER_NAME}] [--fps {FPS}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it wll be set to 30.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.
- `TARGET_LAYER_NAME`: Layer name to generate GradCAM localization map.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `RESIZE_ALGORITHM`: Resize algorithm used for resizing. If not specified, it will be set to `bilinear`.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Get GradCAM results of a I3D model, using a video file as input and then generate an gif file with 10 fps.

    ```shell
    python demo/demo_gradcam.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
        checkpoints/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth demo/demo.mp4 \
        --target-layer-name backbone/layer4/1/relu --fps 10 \
        --out-filename demo/demo_gradcam.gif
    ```

2. Get GradCAM results of a TSM model, using a video file as input and then generate an gif file, loading checkpoint from url.

    ```shell
    python demo/demo_gradcam.py configs/recognition/tsm/tsm_r50_video_inference_1x1x8_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth \
        demo/demo.mp4 --target-layer-name backbone/layer4/1/relu --out-filename demo/demo_gradcam_tsm.gif
    ```

## Webcam demo

We provide a demo script to implement real-time action recognition from web camera. In order to get predict results in range `[0, 1]`, make sure to set `model.['test_cfg'] = dict(average_clips='prob')` in config file.

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

**Note**: If your hardware is good enough, increasing the value of `DRAWING_FPS` and `INFERENCE_FPS` will get a better experience.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth demo/label_map_k400.txt --average-size 5 \
      --threshold 0.2 --device cpu
    ```

2. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      demo/label_map_k400.txt --average-size 5 --threshold 0.2 --device cpu
    ```

3. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth demo/label_map_k400.txt \
      --average-size 5 --threshold 0.2
    ```

**Note:** Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case.
Users can change:

1). `SampleFrames` step (especially the number of `clip_len` and `num_clips`) of `test_pipeline` in the config file, like `--cfg-options data.test.pipeline.0.num_clips=3`.
2). Change to the suitable Crop methods like `TenCrop`, `ThreeCrop`, `CenterCrop`, etc. in `test_pipeline` of the config file, like `--cfg-options data.test.pipeline.4.type=CenterCrop`.
3). Change the number of `--average-size`. The smaller, the faster.

## Long video demo

We provide a demo script to predict different labels using a single long video. In order to get predict results in range `[0, 1]`, make sure to set `test_cfg = dict(average_clips='prob')` in config file.

```shell
python demo/long_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    ${OUT_FILE} [--input-step ${INPUT_STEP}] [--device ${DEVICE_TYPE}] [--threshold ${THRESHOLD}]
```

Optional arguments:

- `OUT_FILE`: Path to the output, either video or json file
- `INPUT_STEP`: Input step for sampling frames, which can help to get more spare input. If not specified , it will be set to 1.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.01.
- `STRIDE`: By default, the demo generates a prediction for each single frame, which might cost lots of time. To speed up, you can set the argument `STRIDE` and then the demo will generate a prediction every `STRIDE x sample_length` frames (`sample_length` indicates the size of temporal window from which you sample frames, which equals to `clip_len x frame_interval`). For example, if the sample_length is 64 frames and you set `STRIDE` to 0.5, predictions will be generated every 32 frames. If set as 0, predictions will be generated for each frame. The desired value of `STRIDE` is (0, 1], while it also works for `STRIDE > 1` (the generated predictions will be too sparse). Default: 0.
- `LABEL_COLOR`: Font Color of the labels in (B, G, R). Default is white, that is (256, 256, 256).
- `MSG_COLOR`: Font Color of the messages in (B, G, R). Default is gray, that is (128, 128, 128).

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Predict different labels in a long video by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO \
      --input-step 3 --device cpu --threshold 0.2
    ```

2. Predict different labels in a long video by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
    ```

3. Predict different labels in a long video from web by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4 \
      demo/label_map_k400.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
    ```

4. Predict different labels in a long video by using a I3D model on gpu, with input_step=1, threshold=0.01 as default and print the labels in cyan.

    ```shell
    python demo/long_video_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth PATH_TO_LONG_VIDEO demo/label_map_k400.txt PATH_TO_SAVED_VIDEO \
      --label-color 255 255 0
    ```

5. Predict different labels in a long video by using a I3D model on gpu and save the results as a `json` file

    ```shell
    python demo/long_video_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth PATH_TO_LONG_VIDEO demo/label_map_k400.txt ./results.json
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
- `LABEL_MAP`: The label map used. Default: `demo/label_map_ava.txt`.
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
    --config configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map demo/label_map_ava.txt \
    --predict-stepsize 40 \
    --output-fps 20 \
    --show
```
