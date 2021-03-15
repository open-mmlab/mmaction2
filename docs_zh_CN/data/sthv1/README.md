# 准备 Something-Something V1

## 简介

```
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

用户可参考该数据集的 [官网](https://20bn.com/datasets/something-something/v1)，以获取数据集相关的基本信息。
在数据集准备前，请确保当前所在文件夹位置为 `$MMACTION2/tools/data/sthv1/`。

## 步骤 1. 准备标注文件

首先，用户需要在 [官网](https://20bn.com/datasets/something-something/v1) 进行注册，才能对标注文件进行下载。下载好的标准文件需要放在 `$MMACTION2/data/sthv1/annotations` 文件夹下。

## 步骤 2. 准备 RGB 帧

因为 [官网](https://20bn.com/datasets/something-something/v1) 并未提供原始视频文件，只提供了对原视频文件进行抽取得到的 RGB 帧，用户可在 [官网](https://20bn.com/datasets/something-something/v1) 直接对其进行下载。

将下载好的 RGB 帧放在 `$MMACTION2/data/sthv1/` 文件夹下，并使用以下脚本进行解压。

```shell
cd $MMACTION2/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION2/tools/data/sthv1/
```

用户可使用以下脚本，对原视频进行裁剪，得到密集编码且更小尺寸的视频。

```
python ../resize_video.py ../../../data/ucf101/videos/ ../../../data/ucf101/videos_256p_dense_cache --dense --level 2 --ext avi
```

## 步骤 3. 抽取视频帧和光流

如果用户只想使用视频进行加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 进行 [denseflow](https://github.com/open-mmlab/denseflow) 的安装。

如果用户有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 存储中。所抽取的视频帧和光流约占据 100 GB 的存储空间。

用户可以运行以下命令在 SSD 中建立软连接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/ucf101_extracted/
ln -s /mnt/SSD/ucf101_extracted/ ../../../data/ucf101/rawframes
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只对 RGB 帧** 进行抽取。

```shell
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 对 RGB 帧进行抽取。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本使用 "tvl1" 算法进行抽取。

```shell
bash extract_frames.sh
```

## 步骤 4. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

## 步骤 5. 检查文件夹结构

在走完完整的 UCF-101 数据集准备流程后，
用户可以获得对应的 RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，UCF-101 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```

关于对 UCF-101 进行训练和验证，可以参考 [基础教程](/docs_zh_CN/getting_started.md)。
