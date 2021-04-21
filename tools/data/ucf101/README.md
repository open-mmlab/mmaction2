# Preparing UCF-101

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/research/data-sets/ucf101/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ucf101/`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
bash download_videos.sh
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_video.py ../../../data/ucf101/videos/ ../../../data/ucf101/videos_256p_dense_cache --dense --level 2 --ext avi
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. The extracted frames (RGB + Flow) will take up about 100GB.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ucf101_extracted/
ln -s /mnt/SSD/ucf101_extracted/ ../../../data/ucf101/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for UCF-101 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for UCF-101.

In the context of the whole project (for UCF-101 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```

For training and evaluating on UCF-101, please refer to [getting_started.md](/docs/getting_started.md).
