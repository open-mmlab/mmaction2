# 准备 ActivityNet

## 简介

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

用户可参考该数据集的 [官网](http://activity-net.org/)，以获取数据集相关的基本信息。
对于时序动作检测任务，用户可以使用这个 [代码库](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) 提供的缩放过（rescaled）的 ActivityNet 特征，
或者使用 MMAction2 进行特征提取（这将具有更高的精度）。MMAction2 同时提供了以上所述的两种数据使用流程。
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/activitynet/`。

## 选项 1：用户可以使用这个 [代码库](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) 提供的特征

### 步骤 1. 下载标注文件

首先，用户可以使用以下命令下载标注文件。

```shell
bash download_feature_annotations.sh
```

### 步骤 2. 准备视频特征

之后，用户可以使用以下命令下载 ActivityNet 特征。

```shell
bash download_features.sh
```

### 步骤 3. 处理标注文件

之后，用户可以使用以下命令处理下载的标注文件，以便于训练和测试。
该脚本会首先合并两个标注文件，然后再将其分为 `train`, `val` 和 `test` 三个部分。

```shell
python process_annotations.py
```

## 选项 2：使用 MMAction2 对 [官网](http://activity-net.org/) 提供的视频进行特征抽取

### 步骤 1. 下载标注文件

首先，用户可以使用以下命令下载标注文件。

```shell
bash download_annotations.sh
```

### 步骤 2. 准备视频

之后，用户可以使用以下脚本准备视频数据。
该代码参考自 [官方爬虫](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)，该过程将会耗费较多时间。

```shell
bash download_videos.sh
```

由于 ActivityNet 数据集中的一些视频已经在 YouTube 失效，[官网](http://activity-net.org/) 在谷歌网盘和百度网盘提供了完整的数据集数据。
如果用户想要获取失效的数据集，则需要填写 [下载页面](http://activity-net.org/download.html) 中提供的 [需求表格](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) 以获取 7 天的下载权限。

MMAction2 同时也提供了 [BSN 代码库](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) 的标注文件的下载步骤。

```shell
bash download_bsn_videos.sh
```

对于这种情况，该下载脚本将在下载后更新此标注文件，以确保每个视频都存在。

### 步骤 3. 抽取 RGB 帧和光流

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

可使用以下命令抽取视频帧和光流。

```shell
bash extract_frames.sh
```

以上脚本将会生成短边 256 分辨率的视频。如果用户想生成短边 320 分辨率的视频（即 320p），或者 340x256 的固定分辨率，用户可以通过改变参数由 `--new-short 256` 至 `--new-short 320`，或者 `--new-width 340 --new-height 256` 进行设置
更多细节可参考 [数据准备指南](/docs_zh_CN/data_preparation.md)

### 步骤 4. 生成用于 ActivityNet 微调的文件列表

根据抽取的帧，用户可以生成视频级别（video-level）或者片段级别（clip-level）的文件列表，其可用于微调 ActivityNet。

```shell
python generate_rawframes_filelist.py
```

### 步骤 5. 在 ActivityNet 上微调 TSN 模型

用户可使用 `configs/recognition/tsn` 目录中的 ActivityNet 配置文件进行 TSN 模型微调。
用户需要使用 Kinetics 相关模型（同时支持 RGB 模型与光流模型）进行预训练。

### 步骤 6. 使用预训练模型进行 ActivityNet 特征抽取

在 ActivityNet 上微调 TSN 模型之后，用户可以使用该模型进行 RGB 特征和光流特征的提取。

```shell
python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth
```

在提取完特征后，用户可以使用后处理脚本整合 RGB 特征和光流特征，生成 `100-t X 400-d` 维度的特征用于时序动作检测。

```shell
python activitynet_feature_postprocessing.py --rgb ../../../data/ActivityNet/rgb_feat --flow ../../../data/ActivityNet/flow_feat --dest ../../../data/ActivityNet/mmaction_feat
```

## 最后一步：检查文件夹结构

在完成所有 ActivityNet 数据集准备流程后，用户可以获得对应的特征文件，RGB + 光流文件，视频文件以及标注文件。

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

关于对 ActivityNet 进行训练和验证，可以参考 [基础教程](/docs_zh_CN/getting_started.md).
