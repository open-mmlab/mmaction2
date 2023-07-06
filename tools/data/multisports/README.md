# Preparing Multisports

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{li2021multisports,
  title={Multisports: A multi-person video dataset of spatio-temporally localized sports actions},
  author={Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13536--13545},
  year={2021}
}
```

For basic dataset information, please refer to the official [project](https://deeperaction.github.io/datasets/multisports.html) and the [paper](https://arxiv.org/abs/2105.07404).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/multisports/`.

## Step 1. Prepare Annotations

First of all, you have to download annotations and videos to `$MMACTION2/data/multisports` on the official [website](https://github.com/MCG-NJU/MultiSports), please also download the Person Boxes and put it to `$MMACTION2/data/multisports`.

## Step 2. Prepare Videos

Before this step, please make sure the folder structure looks like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── multisports
│   |   ├── MultiSports_box.zip
│   |   ├── trainval
│   |   |   ├── aerobic_gymnastics.zip
│   |   |   ├── basketball.zip
│   |   |   ├── multisports_GT.pkl
│   |   |   ├──...
│   |   ├── test
│   |   |   ├── aerobic_gymnastics.zip
│   |   |   ├── basketball.zip
│   |   |   ├──...
```

Then, you can use the following command to uncompress.

```shell
cd $MMACTION2/data/multisports/
unzip MultiSports_box.zip
cd $MMACTION2/data/multisports/trainval
find . -name '*.zip' -exec unzip {} \;
cd $MMACTION2/data/multisports/test
find . -name '*.zip' -exec unzip {} \;
cd $MMACTION2/tools/data/multisports/
```

## Step 3. Convert Annotations

you can run the following script to convert annotations and proposals as we need.

```shell
cd $MMACTION2/tools/data/multisports/
python parse_anno.py
```

## Step 5. Check Directory Structure

After the whole data process, you will get the videos and annotation files for MultiSports.

In the context of the whole project (for MultiSports only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── multisports
│   |   ├── annotations
|   │   |   ├── multisports_dense_proposals_test.recall_96.13.pkl
|   │   |   ├── multisports_dense_proposals_train.recall_96.13.pkl
|   │   |   ├── multisports_dense_proposals_val.recall_96.13.pkl
|   │   |   ├── multisports_GT.pkl
|   │   |   ├── multisports_train.csv
|   │   |   ├── multisports_val.csv
│   |   ├── trainval
│   |   |   ├── aerobic_gymnastics
|   │   |   |   ├── v__wAgwttPYaQ_c001.mp4
|   │   |   |   ├── v__wAgwttPYaQ_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── basketball
|   │   |   |   ├── v_-6Os86HzwCs_c001.mp4
|   │   |   |   ├── v_-6Os86HzwCs_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── multisports_GT.pkl
│   |   |   ├──...
│   |   ├── test
│   |   |   ├── aerobic_gymnastics
|   │   |   |   ├── v_2KroSzspz-c_c001.mp4
|   │   |   |   ├── v_2KroSzspz-c_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── basketball
|   │   |   |   ├── v_1tefH1iPbGM_c001.mp4
|   │   |   |   ├── v_1tefH1iPbGM_c002.mp4
│   |   |   ├──...
```

We don't need the zip files under the project, you can handle them as you want.
For training and evaluating on MultiSports, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).
