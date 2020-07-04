# Preparing Something-Something V1

For basic dataset information, you can refer to the dataset [website](https://20bn.com/datasets/something-something/v1).
Before we start, please make sure that the directory is located at `$MMACTION/tools/data/sthv1/`.

## Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION/data/sthv1/annotations` on the official [website](https://20bn.com/datasets/something-something/v1).

## Step 2. Prepare Videos

Then, you can download all data parts to `$MMACTION/data/sthv1/` and use the following command to extract.

```shell
cd $MMACTION/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION/tools/data/sthv1/
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [dense_flow](https://github.com/innerlee/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv1_extracted/
ln -s /mnt/SSD/sthv1_extracted/ ../../../data/sthv1/rawframes
```

If you didn't install dense_flow in the installation or only want to play with RGB frames (since extracting optical flow can be time-comsuming), consider running the following script to extract **RGB-only** frames.

```shell
cd $MMACTION/tools/data/sthv1/
bash extract_rgb_frames.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION/tools/data/sthv1/
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION/tools/data/sthv1/
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Something-Something V1 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Something-Something V1.

In the context of the whole project (for Something-Something V1 only), the folder structure will look like:

```
mmaction
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 100000.mp4
│   |   |   ├── 100001.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 100000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 100001
│   |   |   ├── ...

```

For training and evaluating on Something-Something V1, please refer to [getting_started.md](/docs/getting_started.md).
