# Preparing AVA

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

For basic dataset information, please refer to the official [website](https://research.google.com/ava/index.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ava/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

This command will download `ava_v2.1.zip` for AVA `v2.1` annotation. If you need the AVA `v2.2` annotation, you can try the following script.

```shell
VERSION=2.2 bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, use the following script to prepare videos. The codes are adapted from the [official crawler](https://github.com/cvdfoundation/ava-dataset).
Note that this might take a long time.

```shell
bash download_videos.sh
```

Or you can use the following command to downloading AVA videos in parallel using a python script.

```shell
bash download_videos_parallel.sh
```

Note that if you happen to have sudoer or have [GNU parallel](https://www.gnu.org/software/parallel/) on your machine,
you can speed up the procedure by downloading in parallel.

```shell
# sudo apt-get install parallel
bash download_videos_gnu_parallel.sh
```

## Step 3. Cut Videos

Cut each video from its 15th to 30th minute and make them at 30 fps.

```shell
bash cut_videos.sh
```

## Step 4. Extract RGB and Flow

Before extracting, please refer to [install.md](/docs/en/get_started/installation.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ava_extracted/
ln -s /mnt/SSD/ava_extracted/ ../data/ava/rawframes/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using ffmpeg by the following script.

```shell
bash extract_rgb_frames_ffmpeg.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

## Step 5. Fetch Proposal Files

The scripts are adapted from FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks).

Run the following scripts to fetch the pre-computed proposal list.

```shell
bash fetch_ava_proposals.sh
```

## Step 6. Folder Structure

After the whole data pipeline for AVA preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for AVA.

In the context of the whole project (for AVA only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate AVA using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ava
│   │   ├── annotations
│   │   |   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_test.FAIR.recall_93.9.pkl
│   │   |   ├── ava_train_v2.1.csv
│   │   |   ├── ava_val_v2.1.csv
│   │   |   ├── ava_train_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_val_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   │   ├── videos
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_15min
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```

For training and evaluating on AVA, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Reference

1. O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014
