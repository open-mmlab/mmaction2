# Preparing Moments in Time

## Introduction

```
@article{monfortmoments,
    title={Moments in Time Dataset: one million videos for event understanding},
    author={Monfort, Mathew and Andonian, Alex and Zhou, Bolei and Ramakrishnan, Kandan and Bargal, Sarah Adel and Yan, Tom and Brown, Lisa and Fan, Quanfu and Gutfruend, Dan and Vondrick, Carl and others},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2019},
    issn={0162-8828},
    pages={1--8},
    numpages={8},
    doi={10.1109/TPAMI.2019.2901464},
}
```

For basic dataset information, you can refer to the dataset [website](http://moments.csail.mit.edu/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/mit/`.

## Step 1. Prepare Annotations and Videos

First of all, you can run the following script to download the videos along with the annotations.

```shell
bash download_data.sh
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_videos.py ../../../data/mit/videos/ ../../../data/mit/videos_256p_dense_cache --dense --level 2
```

## Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mit_extracted/
ln -s /mnt/SSD/mit_extracted/ ../../../data/mit/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Moments in Time.

In the context of the whole project (for Moments in Time only), the folder structure will look like:

```
mmaction2
├── data
│   └── mit
│       ├── annotations
│       │   ├── license.txt
│       │   ├── moments_categories.txt
│       │   ├── README.txt
│       │   ├── trainingSet.csv
│       │   └── validationSet.csv
│       ├── mit_train_rawframe_anno.txt
│       ├── mit_train_video_anno.txt
│       ├── mit_val_rawframe_anno.txt
│       ├── mit_val_video_anno.txt
│       ├── rawframes
│       │   ├── training
│       │   │   ├── adult+female+singing
│       │   │   │   ├── 0P3XG_vf91c_35
│       │   │   │   │   ├── flow_x_00001.jpg
│       │   │   │   │   ├── flow_x_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── flow_y_00001.jpg
│       │   │   │   │   ├── flow_y_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── img_00001.jpg
│       │   │   │   │   └── img_00002.jpg
│       │   │   │   └── yt-zxQfALnTdfc_56
│       │   │   │   │   ├── ...
│       │   │   └── yawning
│       │   │       ├── _8zmP1e-EjU_2
│       │   │       │   ├── ...
│       │   └── validation
│       │   │       ├── ...
│       └── videos
│           ├── training
│           │   ├── adult+female+singing
│           │   │   ├── 0P3XG_vf91c_35.mp4
│           │   │   ├── ...
│           │   │   └── yt-zxQfALnTdfc_56.mp4
│           │   └── yawning
│           │       ├── ...
│           └── validation
│           │   ├── ...
└── mmaction
└── ...

```

For training and evaluating on Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).
