# Preparing Kinetics-400

For basic dataset information, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/kinetics400/`.

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

If you have already have a backup of the kinetics-400 dataset using the download script above,
you only need to replace all whitespaces in the class name for ease of processing either by [detox](http://manpages.ubuntu.com/manpages/bionic/man1/detox.1.html)

```shell
# sudo apt-get install detox
detox -r ../../../data/kinetics400/videos_train/
detox -r ../../../data/kinetics400/videos_val/
```

or running

```shell
bash rename_classnames.sh
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_videos.py ../../../data/kinetics400/videos_train/ ../../../data/kinetics400/videos_train_256p_dense_cache --dense --level 2
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/kinetics400_extracted_train/
ln -s /mnt/SSD/kinetics400_extracted_train/ ../../../data/kinetics400/rawframes_train/
mkdir /mnt/SSD/kinetics400_extracted_val/
ln -s /mnt/SSD/kinetics400_extracted_val/ ../../../data/kinetics400/rawframes_val/
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

These three commands above can generate images with size 340x256, if you want to generate images with short edge 320 (320p),
you can change the args `--new-width 340 --new-height 256` to `--new-short 320`.
More details can be found in [data_preparation](/docs/data_preparation.md)

## Step 4. Generate File List

you can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh
# execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

## Step 5. Folder Structure

After the whole data pipeline for Kinetics-400 preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for Kinetics-400.

In the context of the whole project (for Kinetics-400 only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics-400 using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── kinetics400_train_list_videos.txt
│   │   ├── kinetics400_val_list_videos.txt
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

For training and evaluating on Kinetics-400, please refer to [getting_started](/docs/getting_started.md).
