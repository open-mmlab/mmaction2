# 准备 HMDB51

## 介绍

[DATASET]

```BibTeX
@article{Kuehne2011HMDBAL,
  title={HMDB: A large video database for human motion recognition},
  author={Hilde Kuehne and Hueihan Jhuang and E. Garrote and T. Poggio and Thomas Serre},
  journal={2011 International Conference on Computer Vision},
  year={2011},
  pages={2556-2563}
}
```

有关数据集的基础信息，用户可以参照数据集 [官网](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)。
在开始前，请确保当前工作目录为 `$MMACTION2/tools/data/hmdb51/`。

为运行下面的 bash 脚本，需要安装 `unrar`。用户可运行 `sudo apt-get install unrar` 安装，或参照 [setup](https://github.com/innerlee/setup)，按照指引，运行 [`zzunrar.sh`](https://github.com/innerlee/setup/blob/master/zzunrar.sh) 脚本实现无管理员权限下的简易安装。

## 步骤 1. 准备标注文件

首先，用户可运行下列脚本来准备标注文件。

```shell
bash download_annotations.sh
```

## 步骤 2. 准备视频

接着，用户可运行下列脚本来准备视频

```shell
bash download_videos.sh
```

## 步骤 3. 提取帧和光流

这步是**可选**的，如果用户只打算使用视频加载器。

在提取前，请参照 [MMAction2 安装文档](/docs_zh_CN/install.md) 来安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果用户有大量的固态硬盘空间，那么推荐在固态硬盘上进行帧的提取以获得更好的输入/输出表现。

用户可以运行下列脚本来软连接到固态硬盘的文件存储位置。

```shell
# 执行这两行指令（假设固态硬盘挂载在 "/mnt/SSD/"上）
mkdir /mnt/SSD/hmdb51_extracted/
ln -s /mnt/SSD/hmdb51_extracted/ ../../../data/hmdb51/rawframes
```

如果用户只想使用图像帧来运行模型（因为提取光流非常耗时），可考虑运行下列脚本，使用 denseflow 只提取**图像帧**。

```shell
bash extract_rgb_frames.sh
```

如果用户不想安装 denseflow，也仍然可以运行下列脚本，使用 OpenCV 来提取图像帧，但输出帧的大小只能和原视频的大小一致。

```shell
bash extract_rgb_frames_opencv.sh
```

如果既想提取图像帧，又想提取光流，那么可以运行下列脚本，使用 "tvl1" 算法来提取帧。

```shell
bash extract_frames.sh
```

## 步骤 4. 生成文件列表

用户可以运行下列脚本来生成原始帧和视频的文件列表

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

## 步骤 5. 检查目录结构

在完成 HMDB51 的全部准备过程后，用户会得到 HMDB51 的原始帧（图像帧 + 光流），视频和标注文件。

在整个项目的上下文中（只对 HMDB51 来说），文件夹结构如下所示：

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

有关在 HMDB51 上进行训练和评估，请参照 [基础教程](/docs_zh_CN/getting_started.md)。
