# Preparing Something-Something V2

## Introduction

<!-- [DATASET] -->

```BibTeX
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [website](https://developer.qualcomm.com/software/ai-datasets/something-something).

`````{tabs}

````{group-tab} Download by MIM
MIM supports downloading from OpenDataLab and preprocessing Something-Something V2 dataset with one command line.
```Bash
# install OpenXlab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login
# download and preprocess by MIM
mim download mmaction2 --dataset sthv2
```

````

## Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION2/data/sthv2/annotations` on the official [website](https://20bn.com/datasets/something-something/v2).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/sthv2/`.
## Step 2. Prepare Videos

Then, you can download all data parts to `$MMACTION2/data/sthv2/` and use the following command to uncompress.

```shell
cd $MMACTION2/data/sthv2/
cat 20bn-something-something-v2-?? | tar zx
cd $MMACTION2/tools/data/sthv2/
```

## Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/en/get_started/installation.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv2_extracted/
ln -s /mnt/SSD/sthv2_extracted/ ../../../data/sthv2/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_frames.sh
```

## Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION2/tools/data/sthv2/
bash generate_{rawframes, videos}_filelist.sh
```

````
`````

### Check Directory Structure

After the whole data process for Something-Something V2 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Something-Something V2.

In the context of the whole project (for Something-Something V2 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv2
│   │   ├── sthv2_{train,val}_list_rawframes.txt(Optional)
│   │   ├── sthv2_{train,val}_list_videos.txt
│   │   ├── annotations(Optional)
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes(Optional)
│   |   |   ├── 1
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```

For training and evaluating on Something-Something V2, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).
