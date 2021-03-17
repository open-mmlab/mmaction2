# 准备 ActivityNet

## 简介

[DATASET]

```BibTeX
@article{Heilbron2015ActivityNetAL,
  title={ActivityNet: A large-scale video benchmark for human activity understanding},
  author={Fabian Caba Heilbron and Victor Escorcia and Bernard Ghanem and Juan Carlos Niebles},
  journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={961-970}
}
```

用户可参考该数据集的 [website](http://activity-net.org/)，以获取数据集相关的基本信息。
对于时序动作检测任务，用户可以使用这个 [代码库](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) 提供的经缩放（rescaled）的 ActivityNet 特征，
或者使用 MMAction2 进行特征提取（这将具有更好地精度）。MMAction2 同时提供了以上所述的两种数据使用流程。
在数据集准备前，请确保当前所在文件夹位置为 `$MMACTION2/tools/data/activitynet/`。

## 选项 1：用户可以使用这个 [代码库](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) 提供的特征

### 步骤 1. 下载标注文件

首先，用户可以使用以下命令进行标注文件下载。

```shell
bash download_feature_annotations.sh
```

### 步骤 2. 准备视频特征

之后，用户可以使用以下命令进行 activitynet 特征下载。

```shell
bash download_features.sh
```

### 步骤 3. 处理标注文件

之后，用户可以使用以下命令处理下载的标注文件，以便于训练和测试。
该脚本会首先合并两个标注文件，然后再将其分为 `train`, `val` 和 `test` 三个部分。

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

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Use following scripts to extract both RGB and Flow.

```shell
bash extract_frames.sh
```

The command above can generate images with new short edge 256. If you want to generate images with short edge 320 (320p), or with fix size 340x256, you can change the args `--new-short 256` to `--new-short 320` or `--new-width 340 --new-height 256`.
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

## 最后一步：检查文件夹结构

在走完完整的 ActivityNet 数据集准备流程后，
用户可以获得对应的特征文件，RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，ActivityNet 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet

(若根据选项 1 进行数据处理)
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..

(若根据选项 2 进行数据处理)
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

关于对 ActivityNet 进行训练和验证，可以参考 [基础教程](/docs/getting_started.md).
