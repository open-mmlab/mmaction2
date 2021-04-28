# Preparing Diving48

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{li2018resound,
  title={Resound: Towards action recognition without representation bias},
  author={Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={513--528},
  year={2018}
}
```

For basic dataset information, you can refer to the official dataset [website](http://www.svcl.ucsd.edu/projects/resound/dataset.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/diving48/`.

## Step 1. Prepare Annotations

You can run the following script to download annotations (considering the correctness of annotation files, we only download V2 version here).

```shell
bash download_annotations.sh
```

## Step 2. Prepare Videos

You can run the following script to download videos.

```shell
bash download_videos.sh
```

## Step 3. Prepare RGB and Flow

This part is **optional** if you only want to use the video loader.

The frames provided in official compressed file are not complete. You may need to go through the following extraction steps to get the complete frames.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/diving48_extracted/
ln -s /mnt/SSD/diving48_extracted/ ../../../data/diving48/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Diving48 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Diving48.

In the context of the whole project (for Diving48 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── diving48
│   │   ├── diving48_{train,val}_list_rawframes.txt
│   │   ├── diving48_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   |   ├── Diving48_V2_train.json
│   |   |   ├── Diving48_V2_test.json
│   |   |   ├── Diving48_vocab.json
│   |   ├── videos
│   |   |   ├── _8Vy3dlHg2w_00000.mp4
│   |   |   ├── _8Vy3dlHg2w_00001.mp4
│   |   |   ├── ...
│   |   ├── rawframes
│   |   |   ├── 2x00lRzlTVQ_00000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2x00lRzlTVQ_00001
│   |   |   ├── ...
```

For training and evaluating on Diving48, please refer to [getting_started.md](/docs/getting_started.md).
