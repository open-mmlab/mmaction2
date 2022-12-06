# 教程3：利用现有模型进行推理

## 基于 RGB 的动作识别模型的推理

MMAction2 提供了一个预测单个视频的推理脚本，为了使预测分数都在`[0,1]`之间，请确保在配置文件中设置 `model.cls_head.average_clips = 'prob'`。

```python
python demo/demo.py ${CONFIG_FILE} ${CHECKPOINT_FILE} ${VIDEO_FILE} ${LABEL_FILE} \
    [--device ${DEVICE_TYPE}] [--fps ${FPS}] [--font-scale ${FONT_SCALE}] [--font-color ${FONT_COLOR}] \
    [--target-resolution ${TARGET_RESOLUTION}] [--out-filename ${OUT_FILE}]
```

可选参数：

- `DEVICE_TYPE`: 指定脚本运行设备，支持 cuda 设备（如 `cuda:0`）或 cpu（`cpu`）。默认为 `cuda:0`。
- `FPS`: 使用帧目录作为输入时，代表输入的帧率。默认为 30。
- `FONT_SCALE`: 输出视频上的字体缩放比例。默认为 0.5。
- `FONT_COLOR`: 输出视频上的字体颜色，默认为白色（ `white`）。
- `TARGET_RESOLUTION`: 输出视频的分辨率，如未指定，使用输入视频的分辨率。
- `RESIZE_ALGORITHM`: 缩放视频时使用的插值方法，默认为 `bicubic`。
- `OUT_FILE`: 输出视频的路径，如未指定，则不会生成输出视频。

示例：

以下示例假设用户的当前目录为 `$MMACTION2`，并已经将所需的模型权重文件下载至目录 `checkpoints/` 下，用户也可以使用所提供的 URL 来直接加载模型权重，文件将会被默认下载至 `$HOME/.cache/torch/checkpoints`。

1. 在 cuda 设备上，使用 TSN 模型进行视频识别：

   ```shell
   # demo.mp4 和 label_map_k400.txt 均来自 Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```

2. 在 cuda 设备上，使用 TSN 模型进行视频识别，并利用 URL 加载模型权重文件：

   ```shell
   # demo.mp4 和 label_map_k400.txt 均来自 Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
   ```

3. 使用 TSN 模型进行视频识别，输出 MP4 格式的识别结果：

   ```shell
   # demo.mp4 和 label_map_k400.txt 均来自 Kinetics-400
   python demo/demo.py configs/recognition/tsn/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
       checkpoints/tsn_r50_8xb32-1x1x8-100e_kinetics400-rgb_20220818-2692d16c.pth \
       demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo/demo_out.mp4
   ```

## 基于骨架的动作识别模型的推理

MMAction2 提供了一个推理脚本，用于使用单个视频预测基于骨架的动作识别结果。

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

可选参数：

- `SKELETON_BASED_ACTION_RECOGNITION_CONFIG_FILE`: 基于骨骼的动作识别配置文件的路径。
- `SKELETON_BASED_ACTION_RECOGNITION_CHECKPOINT`: 基于骨骼的动作识别检查点文件路径或者url。
- `HUMAN_DETECTION_CONFIG_FILE`: 人物检测配置文件。
- `HUMAN_DETECTION_CHECKPOINT`: 人物检测模型检查点文件路径或url。
- `HUMAN_DETECTION_SCORE_THRE`: 人物检测阈值，默认为0.9。
- `HUMAN_DETECTION_CATEGORY_ID`: 人物检测的类别id，默认为0。
- `HUMAN_POSE_ESTIMATION_CONFIG_FILE`: 人体姿态估计配置文件的路径（在COCO-Keypoint数据集上训练过）
- `HUMAN_POSE_ESTIMATION_CHECKPOINT`: 人体姿态估计检查点文件路径或者url（在COCO-Keypoint数据集上训练过）
- `LABEL_MAP`: 使用的标签映射，默认为`'tools/data/skeleton/label_map_ntu60.txt'`。
- `DEVICE`: 运行demo要用的设备，允许的值为CUDA设备如`'cuda:0'` 或者 `'cpu'`. 默认为`'cuda:0'`。
- `SHORT_SIDE`: 帧提取时使用的较短边长度，默认为480。

例子：

假设你位于`$MMACTION2`项目中。

1. 使用 Fast-RCNN 作为人体检测器，HRNetw32 作为姿态估计器，PoseC3D-NTURGB+D-60-XSub-Keypoint 作为基于骨骼的动作识别器

   ```shell
   python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
       --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
       --checkpoint https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
       --det-config demo/skeleton_demo_cfg/faster-rcnn_r50_fpn_2x_coco_infer.py \
       --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
       --det-score-thr 0.9 \
       --det-cat-id 0 \
       --pose-config demo/skeleton_demo_cfg/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
       --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
       --label-map tools/data/skeleton/label_map_ntu60.txt
   ```

2. 使用 Fast-RCNN 作为人体检测器，HRNetw32 作为姿态估计器，STGCN-NTURGB+D-60-XSub-Keypoint 作为基于骨骼的动作识别器

   ```shell
   python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo/demo_skeleton_out.mp4 \
       --config configs/skeleton/stgcn/stgcn_1xb16-80e_ntu60-xsub-keypoint.py \
       --checkpoint https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth \
       --det-config demo/skeleton_demo_cfg/faster-rcnn_r50_fpn_2x_coco_infer.py \
       --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
       --det-score-thr 0.9 \
       --det-cat-id 0 \
       --pose-config demo/skeleton_demo_cfg/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
       --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
       --label-map tools/data/skeleton/label_map_ntu60.txt
   ```
