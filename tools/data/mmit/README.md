# Preparing Multi-Moments in Time

## Introduction

<!-- [DATASET] -->

```BibTeX
@misc{monfort2019multimoments,
    title={Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding},
    author={Mathew Monfort and Kandan Ramakrishnan and Alex Andonian and Barry A McNamara and Alex Lascelles, Bowen Pan, Quanfu Fan, Dan Gutfreund, Rogerio Feris, Aude Oliva},
    year={2019},
    eprint={1911.00232},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [website](http://moments.csail.mit.edu).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/mmit/`.

## Step 1. Prepare Annotations and Videos

First of all, you have to visit the official [website](http://moments.csail.mit.edu/), fill in an application form for downloading the dataset. Then you will get the download link. You can use `bash preprocess_data.sh` to prepare annotations and videos. However, the download command is missing in that script. Remember to download the dataset to the proper place follow the comment in this script.

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_videos.py ../../../data/mmit/videos/ ../../../data/mmit/videos_256p_dense_cache --dense --level 2
```

## Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

First, you can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mmit_extracted/
ln -s /mnt/SSD/mmit_extracted/ ../../../data/mmit/rawframes
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

## Step 3. Generate File List

you can run the follow script to generate file list in the format of rawframes or videos.

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

## Step 4. Check Directory Structure

After the whole data process for Multi-Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Multi-Moments in Time.

In the context of the whole project (for Multi-Moments in Time only), the folder structure will look like:

```
mmaction2/
└── data
    └── mmit
        ├── annotations
        │   ├── moments_categories.txt
        │   ├── trainingSet.txt
        │   └── validationSet.txt
        ├── mmit_train_rawframes.txt
        ├── mmit_train_videos.txt
        ├── mmit_val_rawframes.txt
        ├── mmit_val_videos.txt
        ├── rawframes
        │   ├── 0-3-6-2-9-1-2-6-14603629126_5
        │   │   ├── flow_x_00001.jpg
        │   │   ├── flow_x_00002.jpg
        │   │   ├── ...
        │   │   ├── flow_y_00001.jpg
        │   │   ├── flow_y_00002.jpg
        │   │   ├── ...
        │   │   ├── img_00001.jpg
        │   │   └── img_00002.jpg
        │   │   ├── ...
        │   └── yt-zxQfALnTdfc_56
        │   │   ├── ...
        │   └── ...

        └── videos
            └── adult+female+singing
                ├── 0-3-6-2-9-1-2-6-14603629126_5.mp4
                └── yt-zxQfALnTdfc_56.mp4
            └── ...
```

For training and evaluating on Multi-Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).
