# 准备 Diving48

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{li2018resound,
  title={Resound: Towards action recognition without representation bias},
  author={Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={513--528},
  year={2018}
}
```

用户可参考该数据集的 [官网](http://www.svcl.ucsd.edu/projects/resound/dataset.html)，以获取数据集相关的基本信息。

`````{tabs}

````{group-tab} 使用 MIM 下载
# MIM 支持下载 Diving48 数据集。用户可以通过一行命令，从 OpenDataLab 进行下载，并进行预处理。
```Bash
# 安装 OpenXLab CLI 工具
pip install -U openxlab
# 登录 OpenXLab
openxlab login
# 通过 MIM 进行数据集下载，预处理。注意这将花费较长时间
mim download mmaction2 --dataset diving48
```

````

````{group-tab} 从官方源下载
## 步骤 1. 下载标注文件

用户可以使用以下命令下载标注文件（考虑到标注的准确性，这里仅下载 V2 版本）。在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/diving48/`。

```shell
bash download_annotations.sh
```

## 步骤 2. 准备视频

用户可以使用以下命令下载视频。

```shell
bash download_videos.sh
```

## Step 3. 抽取 RGB 帧和光流

如果用户只想使用视频加载训练，则该部分是 **可选项**。

官网提供的帧压缩包并不完整。若想获取完整的数据，可以使用以下步骤解帧。

在抽取视频帧和光流之前，请参考 [安装指南](/docs/zh_cn/get_started/installation.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果拥有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 中。

可以运行以下命令为 SSD 建立软链接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/diving48_extracted/
ln -s /mnt/SSD/diving48_extracted/ ../../../data/diving48/rawframes
```

如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow **只抽取 RGB 帧**。

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames.sh
```

如果用户没有安装 denseflow，则可以运行以下命令使用 OpenCV 抽取 RGB 帧。然而，该方法只能抽取与原始视频分辨率相同的帧。

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames_opencv.sh
```

如果用户想抽取 RGB 帧和光流，则可以运行以下脚本进行抽取。

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_frames.sh
```

## 步骤 4. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

````
`````

### 检查文件夹结构

在完成所有 Diving48 数据集准备流程后，
用户可以获得对应的 RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Diving48 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── diving48
│   │   ├── diving48_{train,val}_list_rawframes.txt
│   │   ├── diving48_{train,val}_list_videos.txt
│   │   ├── annotations（可选）
│   |   |   ├── Diving48_V2_train.json
│   |   |   ├── Diving48_V2_test.json
│   |   |   ├── Diving48_vocab.json
│   |   ├── videos
│   |   |   ├── _8Vy3dlHg2w_00000.mp4
│   |   |   ├── _8Vy3dlHg2w_00001.mp4
│   |   |   ├── ...
│   |   ├── rawframes（可选）
│   |   |   ├── 2x00lRzlTVQ_00000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2x00lRzlTVQ_00001
│   |   |   ├── ...
```

关于对 Diving48 进行训练和验证，请参考 [训练和测试教程](/docs/en/user_guides/train_test.md)。
