# Preparing Kinetics-710

## Introduction

<!-- [DATASET] -->

```BibTeX
@misc{li2022uniformerv2,
      title={UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer},
      author={Kunchang Li and Yali Wang and Yinan He and Yizhuo Li and Yi Wang and Limin Wang and Yu Qiao},
      year={2022},
      eprint={2211.09552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For basic dataset information, please refer to the [paper](https://arxiv.org/pdf/2211.09552.pdf). The scripts can be used for preparing kinetics-710. MMAction2 supports Kinetics-710
dataset as a concat dataset, which means only provides a list of annotation files, and makes use of the original data of Kinetics-400/600/700 dataset. You could refer to the [config](/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_u8_kinetics710-rgb.py)
for details, which also provides a template config about how to use concat dataset in MMAction2.
Before we start, please make sure that the directory is located at `$MMACTION2`.

## Step 1. Download Kinetics 400/600/700

Kinetics-710 is a video benchmark based on Kinetics-400/600/700, which merges the training set of these Kinetics datasets, and deletes the repeated videos according to Youtube IDs. MMAction2 provides an annotation file based on the Kinetics-400/600/700 on [OpenDataLab](https://opendatalab.com/). So we suggest you download Kinetics-400/600/700 first from OpenDataLab by [MIM](https://github.com/open-mmlab/mim).

```shell
# install OpenXlab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login
# download Kinetics-400/600/700, note that this might take a long time.
mim download mmaction2 --dataset kinetics400
mim download mmaction2 --dataset kinetics600
mim download mmaction2 --dataset kinetics700

```

## Step 2. Download Kinetics-710 Annotations

We provide the annotation list of Kinetics-710 corresponding to OpenDataLab version Kinetics, you could download it from aliyun and unzip it to the `$MMACTION2/data/`

```shell
wget -P data https://download.openmmlab.com/mmaction/dataset/kinetics710/annotations.zip
cd data && unzip annotations.zip && cd ..

```

## Step 3. Folder Structure

After the whole data pipeline for Kinetics preparation.
you can get the videos and annotation files for Kinetics-710.

In the context of the whole project (for Kinetics only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── jf7RDuUTrsQ.mp4
│   │   │   ├── ...
│   ├── kinetics600
│   │   ├── videos
│   │   │   ├── vol_00
│   │   │   │   ├── -A5JFdMXB_k_000018_000028.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vol63
│   ├── kinetics700
│   │   ├── videos
│   │   │   ├── vol_00
│   │   │   │   ├── -Paa0R0tQ1w_000009_000019.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vol63
│   ├── kinetics710
│   │   ├── k400_train_list_videos.txt
│   │   ├── k400_val_list_videos.txt
│   │   ├── k600_train_list_videos.txt
│   │   ├── k600_val_list_videos.txt
│   │   ├── k700_train_list_videos.txt
│   │   ├── k700_val_list_videos.txt
```

For training and evaluating on Kinetics, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).
