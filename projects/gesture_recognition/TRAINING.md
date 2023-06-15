In this document, we show how to prepare the training data and train models required for this project.

# Hand detection

## Data Preparation

We use multiple hand pose estimation datasets to generate a hand detection dataset. The circumscribed rectangle of hand key points of is used as the detection bounding box of the hand. In our demo, we use 4 datasets supported from [MMPose](https://github.com/open-mmlab/mmpose): [FreiHAND Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#freihand-dataset), [OneHand10K](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#onehand10k), [RHD Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#rhd-dataset) and [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe). You can find instructions for preparing each dataset from the corresponding link.

To train the hand detection model, you need to install [MMDet](https://github.com/open-mmlab/mmdetection) and move (or link) the above datasets to `$MMDet/data/`. The folder structure should look like this:

```
mmdetection
├── mmdetection
├── docs
├── tests
├── tools
├── configs
|── data
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

We provide a [parse_pose.py](/projects/gesture_recognition/parse_pose.py) file to convert the annotation files of the above pose datasets to a COCO-style detection annotation. Suppose you are at `$MMDet/data`, run the following command and it will generate `hand_det_train.json` and `hand_det_val.json` at `$MMDet/data/hand_det/`

```
python3 $MMAction/projects/gesture_recognition/parse_pose.py
```

The training annotation file combines the above four data sets, and the validation annotation file just uses the OneHand10K validation for a quick verification. You can also add more hand detection datasets to improve performance. Now we are done with data preparation.

## Training and inference

We provide a [config](/projects/gesture_recognition/configs/rtmdet_nano_320-8xb32_multi-dataset-hand.py) to train a [RTMDet](https://arxiv.org/abs/2212.07784) detection model. Suppose you are at `$MMDet`, you can run the follow command to train the hand detection model with 8 GPUs:

```bash
bash tools/dist_train.sh $MMAction/projects/gesture_recognition/configs/rtmdet_nano_320-8xb32_multi-dataset-hand.py 8
```

To see the detection result for a single image, we can use `$MMDet/demo/image_demo.py`. The follow command will do inference on a single [image](/projects/gesture_recognition/demo/hand_det.jpg) (from a video in the [jester dataset](/tools/data/jester)) and the output should be similar to [this image](/projects/gesture_recognition/demo/hand_det_out.jpg).

```bash
python3 $MMDet/demo/image_demo.py $MMAction/projects/gesture_recognition/demo/hand_det.jpg PATH_TO_HAND_DET_CHECKPOINT --out-dir='.'
```

# Pose estimation

We directly use the pose estimation model from MMPose. Please refer to [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/configs/hand_2d_keypoint/rtmpose) for details.

# Gesture recognition

## Data Preparation

We use the [jester dataset](/tools/data/jester)) to train a skeleton based gesture recognition model. Please follow the link to prepare this dataset (in frames).

Once we have the jester dataset, we provide the [extract_keypoint.py](/projects/gesture_recognition/extract_keypoint.py) to extract the hand keypoints for all video frames in the dataset. This step requires the hand detection model and the pose estimation model in the above two stages. Here is an example to extract the keypoints for the dataset. You may need to modify the path to the dataset, configs or checkpoints according to your system.

```bash
ROOT_TO_JESTER='20bn-jester-v1'
POSE_CONFIG='rtmpose-m_8xb32-210e_coco-wholebody-hand-256x256.py'
POSE_CKPT='rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320_20230524-f6ffed6a.pth'
DET_CONFIG='rtmdet-nano_8xb32-300e_multi-dataset-hand-320x320.py'
DET_CKPT='hand-cocktail5-4e-4-bs256-210e-b74fb594_20230320.pth'
python3 -u extract_keypoint.py $ROOT_TO_JESTER \
    --pose_config $POSE_CONFIG --pose_ckpt $POSE_CKPT \
    --det_config $DET_CONFIG --det-ckpt $DET_CKPT
```

The program will generate a `jester.pkl` file in your current directory. Then move this file to `$MMAction`. We will use this file for skeleton based gesture recognition training.

## Training and inference

We provide a [config](/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-16e_jester-keypoint-2d.py) to train a STGCN++ model. Suppose you are at `$MMAction`, you can run the follow command to train the model with 8 GPUs:

```bash
bash tools/dist_train.sh $MMAction/projects/gesture_recognition/configs/stgcnpp_8xb16-joint-u100-80e_jester-keypoint-2d.py 8
```
