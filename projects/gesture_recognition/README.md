# Gesture Recognition

<!-- [ALGORITHM] -->

## Introduction

<!-- [ABSTRACT] -->

In this project, we present a skeleton based pipeline for gesture recognition. The pipeline is three-stage. The first stage consists of a hand detection module that outputs bounding boxes of human hands from video frames. Afterwards, the second stage employs a pose estimation module to generate keypoints of the detected hands. Finally, the third stage utilizes a skeleton-based gesture recognition module to classify hand actions based on the provided hand skeleton. The three-stage pipeline is lightweight and can achieve real-time on CPU devices. In this README, we provide the models and the inference demo for the project. Training data preparation and training scripts are described in [TRAINING.md](/projects/gesture_recognition/TRAINING.md).

## Hand detection stage

Hand detection results on OneHand10K validation dataset

| Config                                                  | Input Size | bbox mAP | bbox mAP 50 | bbox mAP 75 |                         ckpt                          |                         log                          |
| :------------------------------------------------------ | :--------: | :------: | :---------: | :---------: | :---------------------------------------------------: | :--------------------------------------------------: |
| [rtmdet_nano](/projects/gesture_recognition/configs/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320.py) |  320x320   |  0.8100  |   0.9870    |   0.9190    | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320_20230524-f6ffed6a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320.log) |

## Pose estimation stage

Pose estimation results on COCO-WholeBody-Hand validation set

| Config                                                                                                 | Input Size | PCK@0.2 |  AUC  | EPE  |                  ckpt                   |
| :----------------------------------------------------------------------------------------------------- | :--------: | :-----: | :---: | :--: | :-------------------------------------: |
| [rtmpose_m](/projects/gesture_recognition/configs/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.815  | 0.837 | 4.51 | [ckpt](https://download.openmmlab.com/) |

## Gesture recognition stage

Skeleton base gesture recognition results on Jester validation

| Config                                                  | Input Size | Top 1 accuracy | Top 5 accuracy |                          ckpt                          |                          log                          |
| :------------------------------------------------------ | :--------: | :------------: | :------------: | :----------------------------------------------------: | :---------------------------------------------------: |
| [STGCNPP](/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.py) |  100x17x3  |     89.22      |     97.52      | [ckpt](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d_20230524-fffa7ff0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/projects/gesture_recognition/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.log) |
