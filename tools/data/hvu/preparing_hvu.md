# Preparing HVU

For basic dataset information, please refer to the official [project](https://github.com/holistic-video-understanding/HVU-Dataset/) and the [paper](https://arxiv.org/abs/1904.11451).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/hvu/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

Besides, you need to run the following command to parse the tag list of HVU.

```shell
python parse_tag_list.py
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

You can use the following script to extract both RGB and Flow frames.

```shell
bash extract_frames.sh
```

By default, we generate frames with short edge resized to 256.
More details can be found in [data_preparation](/docs/data_preparation.md)

## Step 4. Generate File List

you can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

## Step 5. Folder Structure

After the whole data pipeline for HVU preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for HVU.

In the context of the whole project (for HVU only), the full folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hvu
│   │   ├── hvu_train_video.json
│   │   ├── hvu_val_video.json
│   │   ├── hvu_train.json
│   │   ├── hvu_val.json
│   │   ├── annotations
│   │   ├── videos_train
│   │   │   ├── OLpWTpTC4P8_000570_000670.mp4
│   │   │   ├── xsPKW4tZZBc_002330_002430.mp4
│   │   │   ├── ...
│   │   ├── videos_val
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

For training and evaluating on HVU, please refer to [getting_started](/docs/getting_started.md).
