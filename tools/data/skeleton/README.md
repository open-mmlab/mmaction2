# Preparing Skeleton Dataset

## Introduction

We release the skeleton annotations used in [Revisiting Skeleton-based Action Recognition](). By default, we use [Faster-RCNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) with ResNet50 backbone for human detection and [HRNet-w32](https://github.com/open-mmlab/mmpose/blob/master/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py) for single person pose estimation. For FineGYM, we use Ground-Truth bounding boxes for the athlete instead of detection bounding boxes.

## Prepare Annotations

Currently, we support three datasets: FineGYM, NTU60_XSub and NTU120_XSub. You can execute following scripts to prepare the annotations.

```shell
bash download_annotations.sh ${DATASET}
```

**TODO**:

- [x] FineGYM
- [x] NTU60_XSub
- [x] NTU120_XSub
- [ ] NTU60_XView
- [ ] NTU120_XSet
- [ ] Kinetics
