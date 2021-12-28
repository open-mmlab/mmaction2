# 准备 GYM

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{shao2020finegym,
  title={Finegym: A hierarchical video dataset for fine-grained action understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2616--2625},
  year={2020}
}
```

请参照 [项目主页](https://sdolivia.github.io/FineGym/) 及 [原论文](https://sdolivia.github.io/FineGym/) 以获取数据集基本信息。
MMAction2 当前支持 GYM99 的数据集预处理。
在开始之前，用户需确保当前目录为 `$MMACTION2/tools/data/gym/`。

## 1. 准备标注文件

首先，用户可以使用如下脚本下载标注文件并进行预处理：

```shell
bash download_annotations.sh
```

## 2. 准备视频

用户可以使用以下脚本准备视频，视频准备代码修改自 [ActivityNet 爬虫](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)。
注意这一步骤将花费较长时间。

```shell
bash download_videos.sh
```

## 3. 裁剪长视频至动作级别

用户首先需要使用以下脚本将 GYM 中的长视频依据标注文件裁剪至动作级别。

```shell
python trim_event.py
```

## 4. 裁剪动作视频至分动作级别

随后，用户需要使用以下脚本将 GYM 中的动作视频依据标注文件裁剪至分动作级别。将视频的裁剪分成两个级别可以带来更高的效率（在长视频中裁剪多个极短片段异常耗时）。

```shell
python trim_subaction.py
```

## 5. 提取 RGB 帧和光流

如果用户仅使用 video loader，则可以跳过本步。

在提取之前，请参考 [安装教程](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

用户可使用如下脚本同时抽取 RGB 帧和光流（提取光流时使用 tvl1 算法）：

```shell
bash extract_frames.sh
```

## 6. 基于提取出的分动作生成文件列表

用户可使用以下脚本为 GYM99 生成训练及测试的文件列表：

```shell
python generate_file_list.py
```

## 7. 目录结构

在完整完成 GYM 的数据处理后，将得到帧文件夹（RGB 帧和光流帧），动作视频片段，分动作视频片段以及训练测试所用标注文件。

在整个项目目录下（仅针对 GYM），完整目录结构如下所示：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── gym
|   |   ├── annotations
|   |   |   ├── gym99_train_org.txt
|   |   |   ├── gym99_val_org.txt
|   |   |   ├── gym99_train.txt
|   |   |   ├── gym99_val.txt
|   |   |   ├── annotation.json
|   |   |   └── event_annotation.json
│   │   ├── videos
|   |   |   ├── 0LtLS9wROrk.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw.mp4
│   │   ├── events
|   |   |   ├── 0LtLS9wROrk_E_002407_002435.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006732_006824.mp4
│   │   ├── subactions
|   |   |   ├── 0LtLS9wROrk_E_002407_002435_A_0003_0005.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006244_006252_A_0000_0007.mp4
|   |   └── subaction_frames
```

关于 GYM 数据集上的训练与测试，请参照 [基础教程](/docs_zh_CN/getting_started.md)。
