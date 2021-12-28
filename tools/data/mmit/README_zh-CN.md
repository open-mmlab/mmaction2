# 准备 Multi-Moments in Time

## 简介

<!-- [DATASET] -->

```BibTeX
@misc{monfort2019multimoments,
    title={Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding},
    author={Mathew Monfort and Kandan Ramakrishnan and Alex Andonian and Barry A McNamara and Alex Lascelles, Bowen Pan, Quanfu Fan, Dan Gutfreund, Rogerio Feris, Aude Oliva},
    year={2019},
    eprint={1911.00232},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

用户可以参照数据集 [官网](http://moments.csail.mit.edu/)，获取数据集相关的基本信息。
在准备数据集前，请确保命令行当前路径为 `$MMACTION2/tools/data/mmit/`。

## 步骤 1. Prepare Annotations and Videos

首先，用户需要访问[官网](http://moments.csail.mit.edu/)，填写申请表来下载数据集。
在得到下载链接后，用户可以使用 `bash preprocess_data.sh` 来准备标注文件和视频。
请注意此脚本并没有下载标注和视频文件，用户需要根据脚本文件中的注释，提前下载好数据集，并放/软链接到合适的位置。

为加快视频解码速度，用户需要缩小原视频的尺寸，可使用以下命令获取密集编码版视频：

```
python ../resize_videos.py ../../../data/mmit/videos/ ../../../data/mmit/videos_256p_dense_cache --dense --level 2
```

## Step 2. 抽取帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 上。
用户可使用以下命令为 SSD 建立软链接。

```shell
# 执行这两行指令进行抽取（假设 SSD 挂载在 "/mnt/SSD/"上）
mkdir /mnt/SSD/mmit_extracted/
ln -s /mnt/SSD/mmit_extracted/ ../../../data/mmit/rawframes
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只抽取 RGB 帧**。

```shell
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 抽取 RGB 帧。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本进行抽取。

```shell
bash extract_frames.sh
```

## 步骤 3. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

## 步骤 4. 检查目录结构

在完成 Multi-Moments in Time 数据集准备流程后，用户可以得到 Multi-Moments in Time 的 RGB 帧 + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Multi-Moments in Time 的文件结构如下：

```
mmaction2/
└── data
    └── mmit
        ├── annotations
        │   ├── moments_categories.txt
        │   ├── trainingSet.txt
        │   └── validationSet.txt
        ├── mmit_train_rawframes.txt
        ├── mmit_train_videos.txt
        ├── mmit_val_rawframes.txt
        ├── mmit_val_videos.txt
        ├── rawframes
        │   ├── 0-3-6-2-9-1-2-6-14603629126_5
        │   │   ├── flow_x_00001.jpg
        │   │   ├── flow_x_00002.jpg
        │   │   ├── ...
        │   │   ├── flow_y_00001.jpg
        │   │   ├── flow_y_00002.jpg
        │   │   ├── ...
        │   │   ├── img_00001.jpg
        │   │   └── img_00002.jpg
        │   │   ├── ...
        │   └── yt-zxQfALnTdfc_56
        │   │   ├── ...
        │   └── ...

        └── videos
            └── adult+female+singing
                ├── 0-3-6-2-9-1-2-6-14603629126_5.mp4
                └── yt-zxQfALnTdfc_56.mp4
            └── ...
```

关于对 Multi-Moments in Time 进行训练和验证，可以参照 [基础教程](/docs_zh_CN/getting_started.md)。
