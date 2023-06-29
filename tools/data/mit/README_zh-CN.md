# 准备 Moments in Time

## 简介

<!-- [DATASET] -->

```BibTeX
@article{monfortmoments,
    title={Moments in Time Dataset: one million videos for event understanding},
    author={Monfort, Mathew and Andonian, Alex and Zhou, Bolei and Ramakrishnan, Kandan and Bargal, Sarah Adel and Yan, Tom and Brown, Lisa and Fan, Quanfu and Gutfruend, Dan and Vondrick, Carl and others},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2019},
    issn={0162-8828},
    pages={1--8},
    numpages={8},
    doi={10.1109/TPAMI.2019.2901464},
}
```

用户可以参照数据集 [官网](http://moments.csail.mit.edu/)，获取数据集相关的基本信息。
在准备数据集前，请确保命令行当前路径为 `$MMACTION2/tools/data/mit/`。

## 步骤 1. 准备标注文件和视频文件

首先，用户需要访问[官网](http://moments.csail.mit.edu/)，填写申请表来下载数据集。
在得到下载链接后，用户可以使用 `bash preprocess_data.sh` 来准备标注文件和视频。
请注意此脚本并没有下载标注和视频文件，用户需要根据脚本文件中的注释，提前下载好数据集，并放/软链接到合适的位置。

为加快视频解码速度，用户需要缩小原视频的尺寸，可使用以下命令获取密集编码版视频：

```shell
python ../resize_videos.py ../../../data/mit/videos/ ../../../data/mit/videos_256p_dense_cache --dense --level 2
```

## Step 2. 抽取帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs/zh_cn/get_started/installation.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 上。
用户可使用以下命令为 SSD 建立软链接。

```shell
# 执行这两行指令进行抽取（假设 SSD 挂载在 "/mnt/SSD/"上）
mkdir /mnt/SSD/mit_extracted/
ln -s /mnt/SSD/mit_extracted/ ../../../data/mit/rawframes
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
bash generate_{rawframes, videos}_filelist.sh
```

## 步骤 4. 检查目录结构

在完成 Moments in Time 数据集准备流程后，用户可以得到 Moments in Time 的 RGB 帧 + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Moments in Time 的文件结构如下：

```
mmaction2
├── data
│   └── mit
│       ├── annotations
│       │   ├── license.txt
│       │   ├── moments_categories.txt
│       │   ├── README.txt
│       │   ├── trainingSet.csv
│       │   └── validationSet.csv
│       ├── mit_train_rawframe_anno.txt
│       ├── mit_train_video_anno.txt
│       ├── mit_val_rawframe_anno.txt
│       ├── mit_val_video_anno.txt
│       ├── rawframes
│       │   ├── training
│       │   │   ├── adult+female+singing
│       │   │   │   ├── 0P3XG_vf91c_35
│       │   │   │   │   ├── flow_x_00001.jpg
│       │   │   │   │   ├── flow_x_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── flow_y_00001.jpg
│       │   │   │   │   ├── flow_y_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── img_00001.jpg
│       │   │   │   │   └── img_00002.jpg
│       │   │   │   └── yt-zxQfALnTdfc_56
│       │   │   │   │   ├── ...
│       │   │   └── yawning
│       │   │       ├── _8zmP1e-EjU_2
│       │   │       │   ├── ...
│       │   └── validation
│       │   │       ├── ...
│       └── videos
│           ├── training
│           │   ├── adult+female+singing
│           │   │   ├── 0P3XG_vf91c_35.mp4
│           │   │   ├── ...
│           │   │   └── yt-zxQfALnTdfc_56.mp4
│           │   └── yawning
│           │       ├── ...
│           └── validation
│           │   ├── ...
└── mmaction
└── ...

```

关于对 Moments in Times 进行训练和验证，可以参照 [训练教程](/docs/zh_cn/user_guides/train_test.md)。
