# 准备 HVU

## 简介

<!-- [DATASET] -->

```BibTeX
@article{Diba2019LargeSH,
  title={Large Scale Holistic Video Understanding},
  author={Ali Diba and M. Fayyaz and Vivek Sharma and Manohar Paluri and Jurgen Gall and R. Stiefelhagen and L. Gool},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2019}
}
```

请参照 [官方项目](https://github.com/holistic-video-understanding/HVU-Dataset/) 及 [原论文](https://arxiv.org/abs/1904.11451) 以获取数据集基本信息。
在开始之前，用户需确保当前目录为 `$MMACTION2/tools/data/hvu/`。

## 1. 准备标注文件

首先，用户可以使用如下脚本下载标注文件并进行预处理：

```shell
bash download_annotations.sh
```

此外，用户可使用如下命令解析 HVU 的标签列表：

```shell
python parse_tag_list.py
```

## 2. 准备视频

用户可以使用以下脚本准备视频，视频准备代码修改自 [ActivityNet 爬虫](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)。
注意这一步骤将花费较长时间。

```shell
bash download_videos.sh
```

## 3. 提取 RGB 帧和光流

如果用户仅使用 video loader，则可以跳过本步。

在提取之前，请参考 [安装指南](/docs/zh_cn/get_started/installation.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

用户可使用如下脚本同时抽取 RGB 帧和光流：

```shell
bash extract_frames.sh
```

该脚本默认生成短边长度为 256 的帧，可参考 [数据准备](/docs/zh_cn/user_guides/prepare_dataset.md) 获得更多细节。

## 4. 生成文件列表

用户可以使用以下两个脚本分别为视频和帧文件夹生成文件列表：

```shell
bash generate_videos_filelist.sh
# 为帧文件夹生成文件列表
bash generate_rawframes_filelist.sh
```

## 5. 为每个 tag 种类生成文件列表

若用户需要为 HVU 数据集的每个 tag 种类训练识别模型，则需要进行此步骤。

步骤 4 中生成的文件列表包含不同类型的标签，仅支持使用 HVUDataset 进行涉及多个标签种类的多任务学习。加载数据的过程中需要使用 `LoadHVULabel` 类进行多类别标签的加载，训练过程中使用 `HVULoss` 作为损失函数。

如果用户仅需训练某一特定类别的标签，例如训练一识别模型用于识别 HVU 中 `action` 类别的标签，则建议使用如下脚本为特定标签种类生成文件列表。新生成的列表将只含有特定类别的标签，因此可使用 `VideoDataset` 或 `RawframeDataset` 进行加载。训训练过程中使用 `BCELossWithLogits` 作为损失函数。

以下脚本为类别为 ${category} 的标签生成文件列表，注意仅支持 HVU 数据集包含的 6 种标签类别: action, attribute, concept, event, object, scene。

```shell
python generate_sub_file_list.py path/to/filelist.json ${category}
```

对于类别 ${category}，生成的标签列表文件名中将使用 `hvu_${category}` 替代 `hvu`。例如，若原指定文件名为 `hvu_train.json`，则对于类别 action，生成的文件列表名为 `hvu_action_train.json`。

## 6. 目录结构

在完整完成 HVU 的数据处理后，将得到帧文件夹（RGB 帧和光流帧），视频以及标注文件。

在整个项目目录下（仅针对 HVU），完整目录结构如下所示：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hvu
│   │   ├── hvu_train_video.json
│   │   ├── hvu_val_video.json
│   │   ├── hvu_train.json
│   │   ├── hvu_val.json
│   │   ├── annotations
│   │   ├── videos_train
│   │   │   ├── OLpWTpTC4P8_000570_000670.mp4
│   │   │   ├── xsPKW4tZZBc_002330_002430.mp4
│   │   │   ├── ...
│   │   ├── videos_val
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

关于 HVU 数据集上的训练与测试，请参照 [训练教程](/docs/zh_cn/user_guides/train_test.md)。
