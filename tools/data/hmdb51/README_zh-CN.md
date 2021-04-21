# 准备 HMDB51

## 简介

<!-- [DATASET] -->

```BibTeX
@article{Kuehne2011HMDBAL,
  title={HMDB: A large video database for human motion recognition},
  author={Hilde Kuehne and Hueihan Jhuang and E. Garrote and T. Poggio and Thomas Serre},
  journal={2011 International Conference on Computer Vision},
  year={2011},
  pages={2556-2563}
}
```

用户可以参照数据集 [官网](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)，获取数据集相关的基本信息。
在准备数据集前，请确保命令行当前路径为 `$MMACTION2/tools/data/hmdb51/`。

为运行下面的 bash 脚本，需要安装 `unrar`。用户可运行 `sudo apt-get install unrar` 安装，或参照 [setup](https://github.com/innerlee/setup)，运行 [`zzunrar.sh`](https://github.com/innerlee/setup/blob/master/zzunrar.sh) 脚本实现无管理员权限下的简易安装。

## 步骤 1. 下载标注文件

首先，用户可使用以下命令下载标注文件。

```shell
bash download_annotations.sh
```

## 步骤 2. 下载视频

之后，用户可使用以下指令下载视频

```shell
bash download_videos.sh
```

## 步骤 3. 抽取帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 上。
用户可使用以下命令为 SSD 建立软链接。

```shell
# 执行这两行指令进行抽取（假设 SSD 挂载在 "/mnt/SSD/"上）
mkdir /mnt/SSD/hmdb51_extracted/
ln -s /mnt/SSD/hmdb51_extracted/ ../../../data/hmdb51/rawframes
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只抽取 RGB 帧**。

```shell
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 抽取 RGB 帧。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本，使用 "tvl1" 算法进行抽取。

```shell
bash extract_frames.sh
```

## 步骤 4. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

## 步骤 5. 检查目录结构

在完成 HMDB51 数据集准备流程后，用户可以得到 HMDB51 的 RGB 帧 + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，HMDB51 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hmdb51
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi

│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
│   │   ├── rawframes
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
│   │   │   │   ├── ...
│   │   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1

```

关于对 HMDB51 进行训练和验证，可以参照 [基础教程](/docs_zh_CN/getting_started.md)。
