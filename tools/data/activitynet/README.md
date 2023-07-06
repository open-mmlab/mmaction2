# Preparing ActivityNet

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{Heilbron2015ActivityNetAL,
  title={ActivityNet: A large-scale video benchmark for human activity understanding},
  author={Fabian Caba Heilbron and Victor Escorcia and Bernard Ghanem and Juan Carlos Niebles},
  journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={961-970}
}
```

For basic dataset information, please refer to the official [website](http://activity-net.org/).
For action detection, you can either use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) or extract feature with mmaction2 (which has better performance).
We release both pipeline.
Before we start, please make sure that current working directory is `$MMACTION2/tools/data/activitynet/`.

## Option 1: Use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation)

### Step 1. Download Annotations

First of all, you can run the following script to download annotation files.

```shell
bash download_feature_annotations.sh
```

### Step 2. Prepare Videos Features

Then, you can run the following script to download activitynet features.

```shell
bash download_features.sh
```

### Step 3. Process Annotation Files

Next, you can run the following script to process the downloaded annotation files for training and testing.
It first merges the two annotation files together and then separates the annoations by `train`, `val` and `test`.

```shell
python process_annotations.py
```

## Option 2: Extract ActivityNet feature using MMAction2 with all videos provided in official [website](http://activity-net.org/)

### Step 1. Download Annotations

First of all, you can run the following script to download annotation files.

```shell
bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

Since some videos in the ActivityNet dataset might be no longer available on YouTube, official [website](http://activity-net.org/) has made the full dataset available on Google and Baidu drives.
To accommodate missing data requests, you can fill in this [request form](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) provided in official [download page](http://activity-net.org/download.html) to have a 7-day-access to download the videos from the drive folders.

We also provide download steps for annotations from [BSN repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation)

```shell
bash download_bsn_videos.sh
```

For this case, the downloading scripts update the annotation file after downloading to make sure every video in it exists.

### Step 3. Extract RGB and Flow

Before extracting, please refer to [install.md](/docs/en/get_started/installation.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Use following scripts to extract both RGB and Flow.

```shell
bash extract_frames.sh
```

The command above can generate images with new short edge 256. If you want to generate images with short edge 320 (320p), or with fix size 340x256, you can change the args `--new-short 256` to `--new-short 320` or `--new-width 340 --new-height 256`.
More details can be found in [prepare dataset](/docs/en/user_guides/prepare_dataset.md)

### Step 4. Generate File List for ActivityNet Finetuning

With extracted frames, you can generate video-level or clip-level lists of rawframes, which can be used for ActivityNet Finetuning.

```shell
python generate_rawframes_filelist.py
```

### Step 5. Finetune TSN models on ActivityNet

You can use ActivityNet configs in `configs/recognition/tsn` to finetune TSN models on ActivityNet.
You need to use Kinetics models for pretraining.
Both RGB models and Flow models are supported.

### Step 6. Extract ActivityNet Feature with finetuned ckpts

After finetuning TSN on ActivityNet, you can use it to extract both RGB and Flow feature.

```shell
python ../../misc/clip_feature_extraction.py tsn_extract_rgb_feat_config.py \
  /path/to/rgb_checkpoint.pth ../../../data/ActivityNet/rgb_tarin_feat.pkl \
  --video-list ../../../data/ActivityNet/anet_train_video.txt \
  --video-root ../../../data/ActivityNet/rawframes \
  --dump-score

python ../../misc/clip_feature_extraction.py tsn_extract_rgb_feat_config.py \
  path/to/rgb_checkpoint.pth ../../../data/ActivityNet/rgb_val_feat.pkl \
  --video-list ../../../data/ActivityNet/anet_val_video.txt \
  --video-root ../../../data/ActivityNet/rawframes \
  --dump-score

python ../../misc/clip_feature_extraction.py tsn_extract_flow_feat_config.py \
  /path/to/flow_checkpoint.pth ../../../data/ActivityNet/flow_tarin_feat.pkl \
  --video-list ../../../data/ActivityNet/anet_train_video.txt \
  --video-root ../../../data/ActivityNet/rawframes \
  --dump-score

python ../../misc/clip_feature_extraction.py tsn_extract_flow_feat_config.py \
  /path/to/flow_checkpoint.pth ../../../data/ActivityNet/flow_val_feat.pkl \
  --video-list ../../../data/ActivityNet/anet_val_video.txt \
  --video-root ../../../data/ActivityNet/rawframes \
  --dump-score
```

After feature extraction, you can use our post processing scripts to concat RGB and Flow feature, generate the `100-t X 400-d` feature for Action Detection.

```shell
python activitynet_feature_postprocessing.py --rgb ../../../data/ActivityNet/rgb_feat --flow ../../../data/ActivityNet/flow_feat --dest ../../../data/ActivityNet/mmaction_feat
```

## Final Step. Check Directory Structure

After the whole data pipeline for ActivityNet preparation,
you will get the features, videos, frames and annotation files.

In the context of the whole project (for ActivityNet only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet

(if Option 1 used)
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..

(if Option 2 used)
│   │   ├── anet_train_video.txt
│   │   ├── anet_val_video.txt
│   │   ├── anet_train_clip.txt
│   │   ├── anet_val_clip.txt
│   │   ├── activity_net.v1-3.min.json
│   │   ├── mmaction_feat
│   │   │   ├── v___c8enCfzqw.csv
│   │   │   ├── v___dXUJsj3yo.csv
│   │   │   ├── ..
│   │   ├── rawframes
│   │   │   ├── v___c8enCfzqw
│   │   │   │   ├── img_00000.jpg
│   │   │   │   ├── flow_x_00000.jpg
│   │   │   │   ├── flow_y_00000.jpg
│   │   │   │   ├── ..
│   │   │   ├── ..

```

For training and evaluating on ActivityNet, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).
