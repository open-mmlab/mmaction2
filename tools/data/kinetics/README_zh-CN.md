# 准备 Kinetics-\[400/600/700\]

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

请参照 [官方网站](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) 以获取数据集基本信息。此脚本用于准备数据集 kinetics400，kinetics600，kinetics700。为准备 kinetics 数据集的不同版本，用户需将脚本中的 `${DATASET}` 赋值为数据集对应版本名称，可选项为 `kinetics400`，`kinetics600`， `kinetics700`。
在开始之前，用户需确保当前目录为 `$MMACTION2/tools/data/${DATASET}/`。

**注**：由于部分 YouTube 链接失效，爬取的 Kinetics 数据集大小可能与原版不同。以下是我们所使用 Kinetics 数据集的大小：

|   数据集    | 训练视频 | 验证集视频 |
| :---------: | :------: | :--------: |
| kinetics400 |  240436  |   19796    |

## 1. 准备标注文件

首先，用户可以使用如下脚本从 [Kinetics 数据集官网](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)下载标注文件并进行预处理：

```shell
bash download_annotations.sh ${DATASET}
```

由于部分视频的 URL 不可用，当前官方标注中所含视频数量可能小于初始版本。所以 MMAction2 提供了另一种方式以获取初始版本标注作为参考。
在这其中，Kinetics400 和 Kinetics600 的标注文件来自 [官方爬虫](https://github.com/activitynet/ActivityNet/tree/199c9358907928a47cdfc81de4db788fddc2f91d/Crawler/Kinetics/data)，
Kinetics700 的标注文件于 05/02/2021 下载自 [网站](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)。

```shell
bash download_backup_annotations.sh ${DATASET}
```

## 2. 准备视频

用户可以使用以下脚本准备视频，视频准备代码修改自 [官方爬虫](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics)。注意这一步骤将花费较长时间。

```shell
bash download_videos.sh ${DATASET}
```

**重要提示**：如果在此之前已下载好 Kinetics 数据集的视频，还需使用重命名脚本来替换掉类名中的空格：

```shell
bash rename_classnames.sh ${DATASET}
```

为提升解码速度，用户可以使用以下脚本将原始视频缩放至更小的分辨率（利用稠密编码）：

```bash
python ../resize_videos.py ../../../data/${DATASET}/videos_train/ ../../../data/${DATASET}/videos_train_256p_dense_cache --dense --level 2
```

也可以从 [Academic Torrents](https://academictorrents.com/) 中下载短边长度为 256 的 [kinetics400](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) 和 [kinetics700](https://academictorrents.com/details/49f203189fb69ae96fb40a6d0e129949e1dfec98)，或从 Common Visual Data Foundation 维护的 [cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) 中下载 Kinetics400/Kinetics600/Kinetics-700-2020。

## 3. 提取 RGB 帧和光流

如果用户仅使用 video loader，则可以跳过本步。

在提取之前，请参考 [安装教程](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有足够的 SSD 空间，那么建议将视频抽取为 RGB 帧以提升 I/O 性能。用户可以使用以下脚本为抽取得到的帧文件夹建立软连接：

```shell
# 执行以下脚本 (假设 SSD 被挂载在 "/mnt/SSD/")
mkdir /mnt/SSD/${DATASET}_extracted_train/
ln -s /mnt/SSD/${DATASET}_extracted_train/ ../../../data/${DATASET}/rawframes_train/
mkdir /mnt/SSD/${DATASET}_extracted_val/
ln -s /mnt/SSD/${DATASET}_extracted_val/ ../../../data/${DATASET}/rawframes_val/
```

如果用户只使用 RGB 帧（由于光流提取非常耗时），可以考虑执行以下脚本，仅用 denseflow 提取 RGB 帧：

```shell
bash extract_rgb_frames.sh ${DATASET}
```

如果用户未安装 denseflow，以下脚本可以使用 OpenCV 进行 RGB 帧的提取，但视频原分辨率大小会被保留：

```shell
bash extract_rgb_frames_opencv.sh ${DATASET}
```

如果同时需要 RGB 帧和光流，可使用如下脚本抽帧：

```shell
bash extract_frames.sh ${DATASET}
```

以上的命令生成短边长度为 256 的 RGB 帧和光流帧。如果用户需要生成短边长度为 320 的帧 (320p)，或是固定分辨率为 340 x 256 的帧，可改变参数 `--new-short 256` 为 `--new-short 320` 或 `--new-width 340 --new-height 256`。
更多细节可以参考 [数据准备](/docs_zh_CN/data_preparation.md)。

## 4. 生成文件列表

用户可以使用以下两个脚本分别为视频和帧文件夹生成文件列表：

```shell
bash generate_videos_filelist.sh ${DATASET}
# 为帧文件夹生成文件列表
bash generate_rawframes_filelist.sh ${DATASET}
```

## 5. 目录结构

在完整完成 Kinetics 的数据处理后，将得到帧文件夹（RGB 帧和光流帧），视频以及标注文件。

在整个项目目录下（仅针对 Kinetics），*最简* 目录结构如下所示：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ${DATASET}
│   │   ├── ${DATASET}_train_list_videos.txt
│   │   ├── ${DATASET}_val_list_videos.txt
│   │   ├── annotations
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── abseiling
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── wrapping_present
│   │   │   ├── ...
│   │   │   ├── zumba
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

关于 Kinetics 数据集上的训练与测试，请参照 [基础教程](/docs_zh_CN/getting_started.md)。
