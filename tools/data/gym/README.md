# Preparing GYM

## Introduction

```
@inproceedings{shao2020finegym,
  title={Finegym: A hierarchical video dataset for fine-grained action understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2616--2625},
  year={2020}
}
```

For basic dataset information, please refer to the official [project](https://sdolivia.github.io/FineGym/) and the [paper](https://arxiv.org/abs/2004.06704).
We currently provide the data pre-processing pipeline for GYM99.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/gym/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

## Step 3. Trim Videos into Events.

First, you need to trim long videos into events based on the annotation of GYM with the following scripts.

```shell
python trim_event.py
```

## Step 4. Trim Events into Subactions.

Then, you need to trim events into subactions based on the annotation of GYM with the following scripts. We use the two stage trimming for better efficiency (trimming multiple short clips from a long video can be extremely inefficient, since you need to go over the video many times).

```shell
python trim_subaction.py
```

## Step 5. Extract RGB and Flow

This part is **optional** if you only want to use the video loader for RGB model training.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Run the following script to extract both rgb and flow using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

## Step 6. Generate file list for GYM99 based on extracted subactions.

You can use the following script to generate train / val lists for GYM99.

```shell
python generate_file_list.py
```

## Step 7. Folder Structure

After the whole data pipeline for GYM preparation. You can get the subaction clips, event clips, raw videos and GYM99 train/val lists.

In the context of the whole project (for GYM only), the full folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── gym
|   |   ├── annotations
|   |   |   ├── gym99_train_org.txt
|   |   |   ├── gym99_val_org.txt
|   |   |   ├── gym99_train.txt
|   |   |   ├── gym99_val.txt
|   |   |   ├── annotation.json
|   |   |   └── event_annotation.json
│   │   ├── videos
|   |   |   ├── 0LtLS9wROrk.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw.mp4
│   │   ├── events
|   |   |   ├── 0LtLS9wROrk_E_002407_002435.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006732_006824.mp4
│   │   └── subactions
|   |       ├── 0LtLS9wROrk_E_002407_002435_A_0003_0005.mp4
|   |       ├── ...
|   |       └── zfqS-wCJSsw_E_006244_006252_A_0000_0007.mp4
```

For training and evaluating on GYM, please refer to [getting_started](/docs/getting_started.md).
