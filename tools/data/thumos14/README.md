# Preparing THUMOS'14

## Introduction

```
@misc{THUMOS14,
    author = "Jiang, Y.-G. and Liu, J. and Roshan Zamir, A. and Toderici, G. and Laptev,
    I. and Shah, M. and Sukthankar, R.",
    title = "{THUMOS} Challenge: Action Recognition with a Large
    Number of Classes",
    howpublished = "\url{http://crcv.ucf.edu/THUMOS14/}",
    Year = {2014}
}
```

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/THUMOS14/download.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/thumos14/`.

## Step 1. Prepare Annotations

First of all, run the following script to prepare annotations.

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_videos.sh
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/thumos14_extracted/
ln -s /mnt/SSD/thumos14_extracted/ ../data/thumos14/rawframes/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_frames.sh tvl1
```

.
## Step 4. Fetch File List

This part is **optional** if you do not use SSN model.

You can run the follow script to fetch pre-computed tag proposals.

```shell
cd $MMACTION2/tools/data/thumos14/
bash fetch_tag_proposals.sh
```

## Step 5. Denormalize Proposal File

This part is **optional** if you do not use SSN model.

You can run the follow script to denormalize pre-computed tag proposals according to
actual number of local rawframes.

```shell
cd $MMACTION2/tools/data/thumos14/
bash denormalize_proposal_file.sh
```

## Step 6. Check Directory Structure

After the whole data process for THUMOS'14 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for THUMOS'14.

In the context of the whole project (for THUMOS'14 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── thumos14
│   │   ├── proposals
│   │   |   ├── thumos14_tag_val_normalized_proposal_list.txt
│   │   |   ├── thumos14_tag_test_normalized_proposal_list.txt
│   │   ├── annotations_val
│   │   ├── annotations_test
│   │   ├── videos
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001.mp4
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001.mp4
│   │   │   |   ├── ...
│   │   ├── rawframes
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001
|   │   │   |   │   ├── img_00001.jpg
|   │   │   |   │   ├── img_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_x_00001.jpg
|   │   │   |   │   ├── flow_x_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_y_00001.jpg
|   │   │   |   │   ├── flow_y_00002.jpg
|   │   │   |   │   ├── ...
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001
```

For training and evaluating on THUMOS'14, please refer to [getting_started.md](/docs/getting_started.md).
