# Demo

### Demo link

  * [Video demo](#video-demo): A demo script to predict the recognition result using a single video
  * [Webcam demo](#webcam-demo): A demo script to implement real-time action recognition from web camera

### Video demo

We provide a demo script to predict the recognition result using a single video.

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

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`

1. Recognize a video file as input by using a TSN model on cuda by default.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt
    ```

2. Recognize a list of rawframes as input by using a TSN model on cpu.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --device cpu
    ```

3. Recognize a video file as input by using a TSN model and then generate an mp4 file.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --out-filename demo/demo_out.mp4
    ```

4. Recognize a list of rawframes as input by using a TSN model and then generate a gif file.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --out-filename demo/demo_out.gif
    ```

5. Recognize a video file as input by using a TSN model, then generate an mp4 file with a given resolution and resize algorithm.

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

6. Recognize a video file as input by using a TSN model, then generate an mp4 file with a label in a red color and 10px fontsize.

    ```shell
    # The demo.mp4 and label_map.txt are both from Kinetics-400
    python demo/demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        demo/demo.mp4 demo/label_map.txt --font-size 10 --font-color red \
        --out-filename demo/demo_out.mp4
    ```

7. Recognize a list of rawframes as input by using a TSN model and then generate an mp4 file with 24 fps.

    ```shell
    python demo/demo.py configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py \
        checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
        PATH_TO_FRAMES/ LABEL_FILE --use-frames --fps 24 --out-filename demo/demo_out.gif
    ```

### Webcam demo

We provide a demo script to implement real-time action recognition from web camera.

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

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`

1. Recognize the action from web camera as input by using a TSN model on cpu, averaging the score per 5 times
    and outputting result labels with score higher than 0.2.

    ```shell
    python demo/webcam_demo.py configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py \
      checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth demo/label_map.txt --average-size 5 \
      --threshold 0.2 --device cpu
    ```

2. Recognize the action from web camera as input by using a I3D model on gpu by default, averaging the score per 5 times
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
