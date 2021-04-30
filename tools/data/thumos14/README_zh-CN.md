# 准备 THUMOS'14

## 简介

<!-- [DATASET] -->

```BibTex
@misc{THUMOS14,
    author = {Jiang, Y.-G. and Liu, J. and Roshan Zamir, A. and Toderici, G. and Laptev,
    I. and Shah, M. and Sukthankar, R.},
    title = {{THUMOS} Challenge: Action Recognition with a Large
    Number of Classes},
    howpublished = "\url{http://crcv.ucf.edu/THUMOS14/}",
    Year = {2014}
}
```

用户可以参照数据集 [官网](https://www.crcv.ucf.edu/THUMOS14/download.html)，获取数据集相关的基本信息。
在准备数据集前，请确保命令行当前路径为 `$MMACTION2/tools/data/thumos14/`。

## 步骤 1. 下载标注文件

首先，用户可使用以下命令下载标注文件。

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_annotations.sh
```

## 步骤 2. 下载视频

之后，用户可使用以下指令下载视频

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_videos.sh
```

## 步骤 3. 抽取帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 上。
用户可使用以下命令为 SSD 建立软链接。

```shell
# 执行这两行指令进行抽取（假设 SSD 挂载在 "/mnt/SSD/"上）
mkdir /mnt/SSD/thumos14_extracted/
ln -s /mnt/SSD/thumos14_extracted/ ../data/thumos14/rawframes/
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只抽取 RGB 帧**。

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 抽取 RGB 帧。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本进行抽取。

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_frames.sh tvl1
```

## 步骤 4. 生成文件列表

如果用户不使用 SSN 模型，则该部分是 **可选项**。

可使用运行以下脚本下载预先计算的候选标签。

```shell
cd $MMACTION2/tools/data/thumos14/
bash fetch_tag_proposals.sh
```

## 步骤 5. 去规范化候选文件

如果用户不使用 SSN 模型，则该部分是 **可选项**。

可运行以下脚本，来根据本地原始帧的实际数量，去规范化预先计算的候选标签。

```shell
cd $MMACTION2/tools/data/thumos14/
bash denormalize_proposal_file.sh
```

## 步骤 6. 检查目录结构

在完成 THUMOS'14 数据集准备流程后，用户可以得到 THUMOS'14 的 RGB 帧 + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，THUMOS'14 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── thumos14
│   │   ├── proposals
│   │   |   ├── thumos14_tag_val_normalized_proposal_list.txt
│   │   |   ├── thumos14_tag_test_normalized_proposal_list.txt
│   │   ├── annotations_val
│   │   ├── annotations_test
│   │   ├── videos
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001.mp4
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001.mp4
│   │   │   |   ├── ...
│   │   ├── rawframes
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001
|   │   │   |   │   ├── img_00001.jpg
|   │   │   |   │   ├── img_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_x_00001.jpg
|   │   │   |   │   ├── flow_x_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_y_00001.jpg
|   │   │   |   │   ├── flow_y_00002.jpg
|   │   │   |   │   ├── ...
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001
```

关于对 THUMOS'14 进行训练和验证，可以参照 [基础教程](/docs_zh_CN/getting_started.md)。
