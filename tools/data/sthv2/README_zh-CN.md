# 准备 Something-Something V2

## 简介

<!-- [DATASET] -->

```BibTeX
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

用户可参考该数据集的 [官网](https://developer.qualcomm.com/software/ai-datasets/something-something)，以获取数据集相关的基本信息。

`````{tabs}

````{group-tab} 使用 MIM 下载
# MIM 支持下载 Something-Something V2 数据集。用户可以通过一行命令，从 OpenDataLab 进行下载，并进行预处理。
```Bash
# 安装 OpenXLab CLI 工具
pip install -U openxlab
# 登录 OpenXLab
openxlab login
# 通过 MIM 进行数据集下载，预处理。注意这将花费较长时间
mim download mmaction2 --dataset sthv2
```

````

````{group-tab} 从官方源下载
## 步骤 1. 下载标注文件

首先，用户需要在 [官网](https://developer.qualcomm.com/software/ai-datasets/something-something) 完成注册，才能下载标注文件。下载好的标注文件需要放在 `$MMACTION2/data/sthv2/annotations` 文件夹下。
用户可以使用以下命令下载标注文件。在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/sthv2/`。

## 步骤 2. 准备视频

之后，用户可将下载好的压缩文件放在 `$MMACTION2/data/sthv2/` 文件夹下，并且使用以下指令进行解压。

```shell
cd $MMACTION2/data/sthv2/
cat 20bn-something-something-v2-?? | tar zx
cd $MMACTION2/tools/data/sthv2/
```

## 步骤 3. 抽取 RGB 帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs/zh_cn/get_started/installation.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果拥有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 中。

可以运行以下命令为 SSD 建立软链接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/sthv2_extracted/
ln -s /mnt/SSD/sthv2_extracted/ ../../../data/sthv2/rawframes
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只抽取 RGB 帧**。

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 抽取 RGB 帧。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本进行抽取。

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_frames.sh
```

## 步骤 4. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
cd $MMACTION2/tools/data/sthv2/
bash generate_{rawframes, videos}_filelist.sh
```

````
`````

### 检查文件夹结构

在完成所有 Something-Something V2 数据集准备流程后，
用户可以获得对应的 RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Something-Something V2 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv2
│   │   ├── sthv2_{train,val}_list_rawframes.txt（可选）
│   │   ├── sthv2_{train,val}_list_videos.txt
│   │   ├── annotations（可选）
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes（可选）
│   |   |   ├── 1
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```

关于对 Something-Something V2 进行训练和验证，请参考 [训练和测试教程](/docs/en/user_guides/train_test.md)。
