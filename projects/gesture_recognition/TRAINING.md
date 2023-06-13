# Hand Action Recognition Project

<!-- [ALGORITHM] -->

In this document, we show how to prepare the training data and train models required for this project.

## Hand detection stage

We use multiple hand pose estimation datasets to generate a hand detection dataset. The circumscribed rectangle of hand key points of is used as the detection bouding box of the hand. In our demo, we use 4 datasets supported from [MMPose](https://github.com/open-mmlab/mmpose): [FreiHAND Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#freihand-dataset), [OneHand10K](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#onehand10k), [RHD Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#rhd-dataset) and [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe). You can find instructions for preparing each dataset from the corresponding link.

To train the hand detection model, you need to install [MMDet](https://github.com/open-mmlab/mmdetection) and move (or lint) the above datasets to `$MMDet/data/pose`. The folder structure should look like this:

```
mmdetection
├── mmdetection
├── docs
├── tests
├── tools
├── configs
|── data
    │── pose
        |-- freihand
           │-- annotations
           │-- ..
        |-- onehand10k
           │-- annotations
           │-- ..
        |-- rhd
           │-- annotations
           │-- ..
        │-- halpe
           │-- annotations
           |-- hico_20160224_det
              │-- images
              |-- ..
           │-- ..
```

## Pose estimation stage

## Gesture recognition stage
