# Preparing Kinetics-\[400/600/700\]

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

For basic dataset information, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). The scripts can be used for preparing kinetics400, kinetics600, kinetics700. To prepare different version of kinetics, you need to replace `${DATASET}` in the following examples with the specific dataset name. The choices of dataset names are `kinetics400`, `kinetics600` and `kinetics700`.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/${DATASET}/`.

:::{note}
Because of the expirations of some YouTube links, the sizes of kinetics dataset copies may be different. Here are the sizes of our kinetics dataset copies that used to train all checkpoints.

|   Dataset   | training videos | validation videos |
| :---------: | :-------------: | :---------------: |
| kinetics400 |     240436      |       19796       |

:::

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations by downloading from the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).

```shell
bash download_annotations.sh ${DATASET}
```

Since some video urls are invalid, the number of video items in current official annotations are less than the original official ones.
So we provide an alternative way to download the older one as a reference.
Among these, the annotation files of Kinetics400 and Kinetics600 are from [official crawler](https://github.com/activitynet/ActivityNet/tree/199c9358907928a47cdfc81de4db788fddc2f91d/Crawler/Kinetics/data),
the annotation files of Kinetics700 are from [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) downloaded in 05/02/2021.

```shell
bash download_backup_annotations.sh ${DATASET}
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh ${DATASET}
```

**Important**: If you have already downloaded video dataset using the download script above,
you must replace all whitespaces in the class name for ease of processing by running

```shell
bash rename_classnames.sh ${DATASET}
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```bash
python ../resize_videos.py ../../../data/${DATASET}/videos_train/ ../../../data/${DATASET}/videos_train_256p_dense_cache --dense --level 2
```

You can also download from [Academic Torrents](https://academictorrents.com/) ([kinetics400](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) & [kinetics700](https://academictorrents.com/details/49f203189fb69ae96fb40a6d0e129949e1dfec98) with short edge 256 pixels are available) and [cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) (Host by Common Visual Data Foundation and Kinetics400/Kinetics600/Kinetics-700-2020 are available)

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/${DATASET}_extracted_train/
ln -s /mnt/SSD/${DATASET}_extracted_train/ ../../../data/${DATASET}/rawframes_train/
mkdir /mnt/SSD/${DATASET}_extracted_val/
ln -s /mnt/SSD/${DATASET}_extracted_val/ ../../../data/${DATASET}/rawframes_val/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh ${DATASET}
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh ${DATASET}
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh ${DATASET}
```

The commands above can generate images with new short edge 256. If you want to generate images with short edge 320 (320p), or with fix size 340x256, you can change the args `--new-short 256` to `--new-short 320` or `--new-width 340 --new-height 256`.
More details can be found in [data_preparation](/docs/data_preparation.md)

## Step 4. Generate File List

you can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh ${DATASET}
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh ${DATASET}
```

## Step 5. Folder Structure

After the whole data pipeline for Kinetics preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for Kinetics.

In the context of the whole project (for Kinetics only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ${DATASET}
│   │   ├── ${DATASET}_train_list_videos.txt
│   │   ├── ${DATASET}_val_list_videos.txt
│   │   ├── annotations
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── abseiling
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── wrapping_present
│   │   │   ├── ...
│   │   │   ├── zumba
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

For training and evaluating on Kinetics, please refer to [getting_started](/docs/getting_started.md).
