# Tutorial 3: Inference with existing models

## Inference with Action Recognition Models

MMAction2 provides an inference script to predict the recognition result using a single video. In order to get predict results in range `[0, 1]`, make sure to set `model['cls_head']['average_clips'] = 'prob'` in config file.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--fps ${FPS}] [--font-scale ${FONT_SCALE}] [--font-color ${FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--out-filename ${OUT_FILE}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `cuda:0` or `cpu`. If not specified, it will be set to `cuda:0`.
- `FPS`: FPS value of the output video. If not specified, it will be set to 30.
- `FONT_SCALE`: Font scale of the label added in the video. If not specified, it will be 0.5.
- `FONT_COLOR`: Font color of the label added in the video. If not specified, it will be `white`.
- `TARGET_RESOLUTION`: Resolution(desired_width, desired_height) for resizing the frames before output when using a video as input. If not specified, it will be None and the frames are resized by keeping the existing aspect ratio.
- `OUT_FILE`: Path to the output file which can be a video format or gif format. If not specified, it will be set to `None` and does not generate the output file.

Examples:

Assume that you are located at `$MMACTION2` and have already downloaded the checkpoints to the directory `checkpoints/`,
or use checkpoint url from to directly load corresponding checkpoint, which will be automatically saved in `$HOME/.cache/torch/checkpoints`.

1. Recognize a video file as input by using a TSN model on cuda by default.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```

2. Recognize a video file as input by using a TSN model on cuda by default, loading checkpoint from url.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```
