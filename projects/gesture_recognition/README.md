# Gesture Recognition

<!-- [ALGORITHM] -->

## Introduction

<!-- [ABSTRACT] -->

In this project, we present a skeleton based pipeline for gesture recognition. The pipeline is three-stage. The first stage consists of a hand detection module that outputs bounding boxes of human hands from video frames. Afterwards, the second stage employs a pose estimation module to generate keypoints of the detected hands. Finally, the third stage utilizes a skeleton-based gesture recognition module to classify hand actions based on the provided hand skeleton. The three-stage pipeline is lightweight and can achieve real-time on CPU devices. In this README, we provide the models and the inference demo for the project. Training data preparation and training scripts are described in [TRAINING.md](/projects/gesture_recognition/TRAINING.md). Deployment tools are described in [DEPLOYMENT.md](/projects/gesture_recognition/DEPLOYMENT.md)

## Hand detection stage

Hand detection results on OneHand10K validation dataset

| Arch                                                                                | Input Size | bbox mAP | bbox mAP 50 | bbox mAP 75 |                  ckpt                   |                  log                   |
| :---------------------------------------------------------------------------------- | :--------: | :------: | :---------: | :---------: | :-------------------------------------: | :------------------------------------: |
| [rtmpose_nano](/projects/gesture_recognition/configs/rtmdet_nano_320-8xb32_multi-dataset-hand.py) |  320x320   |  0.8100  |   0.9870    |   0.9190    | [ckpt](https://download.openmmlab.com/) | [log](https://download.openmmlab.com/) |

## Pose estimation stage

Pose estimation results on COCO-WholeBody-Hand validation set

| Arch                                                                                              | Input Size | PCK@0.2 |  AUC  | EPE  |                  ckpt                   |                  log                   |
| :------------------------------------------------------------------------------------------------ | :--------: | :-----: | :---: | :--: | :-------------------------------------: | :------------------------------------: |
| [rtmpose_m](/projects/gesture_recognition/configs/rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py) |  256x256   |  0.815  | 0.837 | 4.51 | [ckpt](https://download.openmmlab.com/) | [log](https://download.openmmlab.com/) |

## Gesture recognition stage
