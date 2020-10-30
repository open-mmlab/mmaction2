# Preparing Something-Something V1

## Introduction

```
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [website](https://20bn.com/datasets/something-something/v1).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/sthv1/`.

## Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION2/data/sthv1/annotations` on the official [website](https://20bn.com/datasets/something-something/v1).

## Step 2. Prepare RGB Frames

Since the [sthv1 website](https://20bn.com/datasets/something-something/v1) doesn't provide the original video data and only extracted RGB frames are available, you have to directly download RGB frames from [sthv1 website](https://20bn.com/datasets/something-something/v1).

You can download all RGB frame parts on [sthv1 website](https://20bn.com/datasets/something-something/v1) to `$MMACTION2/data/sthv1/` and use the following command to extract.

```shell
cd $MMACTION2/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION2/tools/data/sthv1/
```

For users who only want to use RGB frames, you can skip to step 5 to generate file lists in the format of rawframes. Since the prefix of official JPGs is "%05d.jpg" (e.g., "00001.jpg"), we add "filename_tmpl='{:05}.jpg'" to the dict of `data.train`, `data.val` and `data.test` in the config files related with sthv1 like this:

```
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
```

## Step 3. Extract Flow

This part is **optional** if you only want to use RGB frames.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv1_extracted/
ln -s /mnt/SSD/sthv1_extracted/ ../../../data/sthv1/rawframes
```

Then, you can run the following script to extract optical flow based on RGB frames.

```shell
cd $MMACTION2/tools/data/sthv1/
bash extract_flow.sh
```

## Step 4. Encode Videos

This part is **optional** if you only want to use RGB frames.

You can run the following script to encode videos.

```shell
cd $MMACTION2/tools/data/sthv1/
bash encode_videos.sh
```

## Step 5. Generate File List

You can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION2/tools/data/sthv1/
bash generate_{rawframes, videos}_filelist.sh
```

## Step 5. Check Directory Structure

After the whole data process for Something-Something V1 preparation,
you will get the rawframes (RGB + Flow), and annotation files for Something-Something V1.

In the context of the whole project (for Something-Something V1 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── 00001.jpg
│   |   |   |   ├── 00002.jpg
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

For training and evaluating on Something-Something V1, please refer to [getting_started.md](/docs/getting_started.md).
