# Preparing ActivityNet

## Introduction

```
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

## Step 1. Download Annotations
First of all, you can run the following script to download annotation files.
```shell
bash download_annotations.sh
```

## Option 1: Use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation)

### Step 2. Prepare Videos Features
Then, you can run the following script to download activitynet features.
```shell
bash download_features.sh
```

### Step 3. Process Annotation Files
Next, you can run the following script to process the downloaded annotation files for training and testing.
It first merges the two annotation files together and then seperates the annoations by `train`, `val` and `test`.

```shell
python process_annotations.py
```

## Option 2: Extract ActivityNet feature using MMAction2.

### Step 2. Prepare Videos.
Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.
Some videos in the ActivityNet dataset might be no longer available on YouTube, so that after video downloading, the downloading scripts update the annotation file to make sure every video in it exists.

```shell
bash download_videos.sh
```

### Step 3. Extract RGB and Flow
Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Use following scripts to extract both RGB and Flow.

```shell
bash extract_frames.sh
```

These three commands above can generate images with size 340x256, if you want to generate images with short edge 320 (320p),
you can change the args `--new-width 340 --new-height 256` to `--new-short 320`.
More details can be found in [data_preparation](/docs/data_preparation.md)

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
python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth
```

After feature extraction, you can use our post processing scripts to concat RGB and Flow feature, generate the 100-t X 400-d feature for Action Detection.

```shell
python activitynet_feature_postprocessing.py --rgb ../../../data/ActivityNet/rgb_feat --flow ../../../data/ActivityNet/flow_feat --dest ../../../data/ActivityNet/mmaction_feat
```

## Final Step. Check Directory Structure

After the whole data pipeline for ActivityNet preparation,
you will get the features and annotation files.

In the context of the whole project (for ActivityNet only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv

(if Option 1 used)
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..

(if Option 2 used)
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
│   │   ├── anet_train_video.txt
│   │   ├── anet_val_video.txt
│   │   ├── anet_train_clip.txt
│   │   ├── anet_val_clip.txt
```

For training and evaluating on ActivityNet, please refer to [getting_started.md](/docs/getting_started.md).
