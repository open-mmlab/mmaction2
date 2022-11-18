# Tutorial 3: Inference with existing models

## Inference with RGB-based Action Recognition Models

MMAction2 provides an inference script to predict the recognition result using a single video. In order to get predict results in range `[0, 1]`, make sure to set `model.cls_head.average_clips = 'prob'` in config file.

```shell
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--fps ${FPS}] [--font-scale ${FONT_SCALE}] [--font-color ${FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--out-filename ${OUT_FILE}]
```

Optional arguments:

- `DEVICE_TYPE`: Type of device to run the demo. Allowed values are cuda device like `'cuda:0'` or `'cpu'`. Defaults to `'cuda:0'`.
- `FPS`: FPS value of the output video. Defaults to 30.
- `FONT_SCALE`: Font scale of the label added in the video. Defaults to 0.5.
- `FONT_COLOR`: Font color of the label added in the video. Defaults to `'white'`.
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

3. Recognize a video file as input by using a TSN model and then generate an mp4 file.

   ```shell
   # The demo.mp4 and label_map_k400.txt are both from Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo/demo_out.mp4
   ```

## Inference with Skeleton-based Action Recognition Models

MMAction2 provides an inference script to predict the skeleton-based action recognition result using a single video.

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
    --config configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py \
    --checkpoint https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
```
