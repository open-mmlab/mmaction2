# Demo

## Demo link

- [Video demo](#video-demo): A demo script to predict the recognition result using a single video
- [Video GradCAM Demo](#video-gradcam-demo): A demo script to visualize GradCAM results using a single video.
- [Webcam demo](#webcam-demo): A demo script to implement real-time action recognition from web camera
- [Long Video demo](#long-video-demo): a demo script to predict different labels using a single long video.

## Video demo

We provide a demo script to predict the recognition result using a single video. In order to get predict results in range `[0, 1]`, make sure to set `test_cfg = dict(average_clips='prob')` in config file.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} {LABEL_FILE} [--use-frames] \
    [--device ${DEVICE_TYPE}] [--fps {FPS}] [--font-size {FONT_SIZE}] [--font-color {FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--resize-algorithm {RESIZE_ALGORITHM}] [--out-filename {OUT_FILE}]
```

Optional arguments:

- `--use-frames`: If specified, the demo will take rawframes as input. Otherwise, it will take a video as input.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video when using rawframes as input. If not specified, it wll be set to 30.
- `FONT_SIZE`: Font size of the label added in the video. If not specified, it wll be set to 20.
- `FONT_COLOR`: Font color of the label added in the video. If not specified, it will be `white`.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `RESIZE_ALGORITHM`: Resize algorithm used for resizing. If not specified, it will be set to `bicubic`.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cahe/torch/checkpoints`.

1. Recognize a video file as input by using a TSN model on cuda by default.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt
    ```

2. Recognize a video file as input by using a TSN model on cuda by default, loading checkpoint from url.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt
    ```

3. Recognize a list of rawframes as input by using a TSN model on cpu.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --device cpu
    ```

4. Recognize a video file as input by using a TSN model and then generate an mp4 file.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --out-filename demo/demo_out.mp4
    ```

5. Recognize a list of rawframes as input by using a TSN model and then generate a gif file.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --out-filename demo/demo_out.gif
    ```

6. Recognize a video file as input by using a TSN model, then generate an mp4 file with a given resolution and resize algorithm.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --target-resolution 340 256 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    # If either dimension is set to -1, the frames are resized by keeping the existing aspect ratio
    # For --target-resolution 170 -1, original resolution (340, 256) -> target resolution (170, 128)
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --target-resolution 170 -1 --resize-algorithm bilinear \
        --out-filename demo/demo_out.mp4
    ```

7. Recognize a video file as input by using a TSN model, then generate an mp4 file with a label in a red color and 10px fontsize.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --font-size 10 --font-color red \
        --out-filename demo/demo_out.mp4
    ```

8. Recognize a list of rawframes as input by using a TSN model and then generate an mp4 file with 24 fps.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --fps 24 --out-filename demo/demo_out.gif
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
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cahe/torch/checkpoints`.

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

We provide a demo script to implement real-time action recognition from web camera. In order to get predict results in range `[0, 1]`, make sure to set `test_cfg = dict(average_clips='prob')` in config file.

```shell
python demo/webcam_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--camera-id ${CAMERA_ID}] [--threshold ${THRESHOLD}] \
    [--average-size ${AVERAGE_SIZE}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `CAMERA_ID`: ID of camera device If not specified, it will be set to 0.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.
- `AVERAGE_SIZE`: Number of latest clips to be averaged for prediction. If not specified, it will be set to 1.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cahe/torch/checkpoints`.

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth demo/label_map.txt --average-size 5 \
      --threshold 0.2 --device cpu
    ```

2. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      demo/label_map.txt --average-size 5 --threshold 0.2 --device cpu
    ```

3. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth demo/label_map.txt \
      --average-size 5 --threshold 0.2
    ```

**Note:** Considering the efficiency difference for users' hardware, Some modifications might be done to suit the case.
Users can change:

1). `SampleFrames` step (especially the number of `clip_len` and `num_clips`) of `test_pipeline` in the config file.
2). Change to the suitable Crop methods like `TenCrop`, `ThreeCrop`, `CenterCrop`, etc. in `test_pipeline` of the config file.
3). Change the number of `--average-size`. The smaller, the faster.

## Long video demo

We provide a demo script to predict different labels using a single long video. In order to get predict results in range `[0, 1]`, make sure to set `test_cfg = dict(average_clips='prob')` in config file.

```shell
python demo/long_video_demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    ${OUT_FILE} [--input-step ${INPUT_STEP}] [--device ${DEVICE_TYPE}] [--threshold ${THRESHOLD}]
```

Optional arguments:

- `OUT_FILE`: Path to the output video file.
- `INPUT_STEP`: Input step for sampling frames, which can help to get more spare input. If not specified , it will be set to 1.
- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `THRESHOLD`: Threshold of prediction score for action recognition. Only label with score higher than the threshold will be shown. If not specified, it will be set to 0.01.
- `STRIDE`: By default, the demo generates a prediction for each single frame, which might cost lots of time. To speed up, you can set the argument `STRIDE` and then the demo will generate a prediction every `STRIDE x sample_length` frames (`sample_length` indicates the size of temporal window from which you sample frames, which equals to `clip_len x frame_interval`). For example, if the sample_length is 64 frames and you set `STRIDE` to 0.5, predictions will be generated every 32 frames. If set as 0, predictions will be generated for each frame. The desired value of `STRIDE` is (0, 1], while it also works for `STRIDE > 1` (the generated predictions will be too sparse). Default: 0.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from `configs/` to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cahe/torch/checkpoints`.

1. Predict different labels in a long video by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth PATH_TO_LONG_VIDEO demo/label_map.txt PATH_TO_SAVED_VIDEO \
      --input-step 3 --device cpu --threshold 0.2
    ```

2. Predict different labels in a long video by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      PATH_TO_LONG_VIDEO demo/label_map.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
    ```

3. Predict different labels in a long video from web by using a TSN model on cpu, with 3 frames for input steps (that is, random sample one from each 3 frames)
   and outputting result labels with score higher than 0.2, loading checkpoint from url.

    ```shell
    python demo/long_video_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
      https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4 \
      demo/label_map.txt PATH_TO_SAVED_VIDEO --input-step 3 --device cpu --threshold 0.2
    ```

4. Predict different labels in a long video by using a I3D model on gpu, with input_step=1 and threshold=0.01 as default.

    ```shell
    python demo/long_video_demo.py configs/recognition/i3d/i3d_r50_video_inference_32x2x1_100e_kinetics400_rgb.py \
      checkpoints/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth PATH_TO_LONG_VIDEO demo/label_map.txt PATH_TO_SAVED_VIDEO \
    ```
