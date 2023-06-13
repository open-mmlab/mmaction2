In this document, we show how to prepare the training data and train models required for this project.

# Hand detection

## Data Preparation

We use multiple hand pose estimation datasets to generate a hand detection dataset. The circumscribed rectangle of hand key points of is used as the detection bounding box of the hand. In our demo, we use 4 datasets supported from [MMPose](https://github.com/open-mmlab/mmpose): [FreiHAND Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#freihand-dataset), [OneHand10K](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#onehand10k), [RHD Dataset](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_hand_keypoint.html#rhd-dataset) and [Halpe](https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_wholebody_keypoint.html#halpe). You can find instructions for preparing each dataset from the corresponding link.

To train the hand detection model, you need to install [MMDet](https://github.com/open-mmlab/mmdetection) and move (or lint) the above datasets to `$MMDet/data/`. The folder structure should look like this:

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

We provide a [parse_pose.py](projects/gesture_recognition/parse_pose.py) file to convert the annotation files of the above pose datasets to a COCO-style detection annotation. Suppose you are at `$MMDet/data`, run the following command and it will generate `hand_det_train.json` and `hand_det_val.json` at `$MMDet/data/hand_det/`

```
python3 $MMAction/projects/gesture_recognition/parse_pose.py
```

The training annotation file combines the above four data sets, and the validation annotation file just uses the OneHand10K validation for a quick verification. You can also add more hand detection datasets to improve performance. Now we are done with data preparation.

## Training and inference

We provide a [config](projects/gesture_recognition/configs/rtmdet_nano_320-8xb32_multi-dataset-hand.py) to train a RTMDet detection model. Suppose you are at `$MMDet`, you can run the follow command to train the hand detection model with 8 GPUs:

```bash
bash tools/dist_train.sh $MMAction/projects/gesture_recognition/configs/rtmdet_nano_320-8xb32_multi-dataset-hand.py 8
```

To see the detection result for a single image, we can use `$MMDet/demo/image_demo.py`. The follow command will do inference on [this image](projects/gesture_recognition/demo/hand_det.jpg) and the output should be similar to [this image](projects/gesture_recognition/demo/hand_det_out.jpg)/

```bash
python3 $MMDet/demo/image_demo.py $MMAction/projects/gesture_recognition/demo/hand_det.jpg PATH_TO_HAND_DET_CHECKPOINT --out-dir='.'
```

# Pose estimation stage

# Gesture recognition stage
