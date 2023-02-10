# Demo

## Outline

- [Modify configs through script arguments](#modify-config-through-script-arguments): Tricks to directly modify configs through script arguments.
- [Video demo](#video-demo): A demo script to predict the recognition result using a single video.
- [Video GradCAM Demo](#video-gradcam-demo): A demo script to visualize GradCAM results using a single video.
- [Webcam demo](#webcam-demo): A demo script to implement real-time action recognition from a web camera.
- [Skeleton-based Action Recognition Demo](#skeleton-based-action-recognition-demo): A demo script to predict the skeleton-based action recognition result using a single video.
- [SpatioTemporal Action Detection Video Demo](#spatiotemporal-action-detection-video-demo): A demo script to predict the spatiotemporal action detection result using a single video.
- [Inferencer Demo](#inferencer): A demo script to implement fast predict for video analysis tasks based on unified inferencer interface.

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
    --config configs/detection/ava/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
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
       --rec configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
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
