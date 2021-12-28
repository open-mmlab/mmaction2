# 准备 AVA

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

请参照 [官方网站](https://research.google.com/ava/index.html) 以获取数据集基本信息。
在开始之前，用户需确保当前目录为 `$MMACTION2/tools/data/ava/`。

## 1. 准备标注文件

首先，用户可以使用如下脚本下载标注文件并进行预处理：

```shell
bash download_annotations.sh
```

这一命令将下载 `ava_v2.1.zip` 以得到 AVA v2.1 标注文件。如用户需要 AVA v2.2 标注文件，可使用以下脚本：

```shell
VERSION=2.2 bash download_annotations.sh
```

## 2. 下载视频

用户可以使用以下脚本准备视频，视频准备代码修改自 [官方爬虫](https://github.com/cvdfoundation/ava-dataset)。
注意这一步骤将花费较长时间。

```shell
bash download_videos.sh
```

亦可使用以下脚本，使用 python 并行下载 AVA 数据集视频：

```shell
bash download_videos_parallel.sh
```

## 3. 截取视频

截取每个视频中的 15 到 30 分钟，设定帧率为 30。

```shell
bash cut_videos.sh
```

## 4. 提取 RGB 帧和光流

在提取之前，请参考 [安装教程](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有足够的 SSD 空间，那么建议将视频抽取为 RGB 帧以提升 I/O 性能。用户可以使用以下脚本为抽取得到的帧文件夹建立软连接：

```shell
# 执行以下脚本 (假设 SSD 被挂载在 "/mnt/SSD/")
mkdir /mnt/SSD/ava_extracted/
ln -s /mnt/SSD/ava_extracted/ ../data/ava/rawframes/
```

如果用户只使用 RGB 帧（由于光流提取非常耗时），可执行以下脚本使用 denseflow 提取 RGB 帧：

```shell
bash extract_rgb_frames.sh
```

如果用户未安装 denseflow，可执行以下脚本使用 ffmpeg 提取 RGB 帧：

```shell
bash extract_rgb_frames_ffmpeg.sh
```

如果同时需要 RGB 帧和光流，可使用如下脚本抽帧：

```shell
bash extract_frames.sh
```

## 5. 下载 AVA 上人体检测结果

以下脚本修改自 [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks)。

可使用以下脚本下载 AVA 上预先计算的人体检测结果：

```shell
bash fetch_ava_proposals.sh
```

## 6. 目录结构

在完整完成 AVA 的数据处理后，将得到帧文件夹（RGB 帧和光流帧），视频以及标注文件。

在整个项目目录下（仅针对 AVA），*最简* 目录结构如下所示：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ava
│   │   ├── annotations
│   │   |   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_test.FAIR.recall_93.9.pkl
│   │   |   ├── ava_train_v2.1.csv
│   │   |   ├── ava_val_v2.1.csv
│   │   |   ├── ava_train_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_val_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   │   ├── videos
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_15min
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```

关于 AVA 数据集上的训练与测试，请参照 [基础教程](/docs_zh_CN/getting_started.md)。
