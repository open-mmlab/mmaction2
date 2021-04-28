# 准备 Jester

## 简介

<!-- [DATASET] -->

```BibTeX
@InProceedings{Materzynska_2019_ICCV,
  author = {Materzynska, Joanna and Berger, Guillaume and Bax, Ingo and Memisevic, Roland},
  title = {The Jester Dataset: A Large-Scale Video Dataset of Human Gestures},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```

用户可以参照数据集 [官网](https://20bn.com/datasets/jester/v1)，获取数据集相关的基本信息。
在准备数据集前，请确保命令行当前路径为 `$MMACTION2/tools/data/jester/`。

## 步骤 1. 下载标注文件

首先，用户需要在 [官网](https://20bn.com/datasets/jester/v1) 完成注册，才能下载标注文件。下载好的标注文件需要放在 `$MMACTION2/data/jester/annotations` 文件夹下。

## 步骤 2. 准备 RGB 帧

[jester 官网](https://20bn.com/datasets/jester/v1) 并未提供原始视频文件，只提供了对原视频文件进行抽取得到的 RGB 帧，用户可在 [jester 官网](https://20bn.com/datasets/jester/v1) 直接下载。

将下载好的压缩文件放在 `$MMACTION2/data/jester/` 文件夹下，并使用以下脚本进行解压。

```shell
cd $MMACTION2/data/jester/
cat 20bn-jester-v1-?? | tar zx
cd $MMACTION2/tools/data/jester/
```

如果用户只想使用 RGB 帧，则可以跳过中间步骤至步骤 5 以直接生成视频帧的文件列表。
由于官网的 JPG 文件名形如 "%05d.jpg" （比如，"00001.jpg"），需要在配置文件的 `data.train`, `data.val` 和 `data.test` 处添加 `"filename_tmpl='{:05}.jpg'"` 代码，以修改文件名模板。

```python
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
```

## 步骤 3. 抽取光流

如果用户只想使用 RGB 帧训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs_zh_CN/install.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果拥有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 中。

可以运行以下命令为 SSD 建立软链接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/jester_extracted/
ln -s /mnt/SSD/jester_extracted/ ../../../data/jester/rawframes
```

如果想抽取光流，则可以运行以下脚本从 RGB 帧中抽取出光流。

```shell
cd $MMACTION2/tools/data/jester/
bash extract_flow.sh
```

## 步骤 4: 编码视频

如果用户只想使用 RGB 帧训练，则该部分是 **可选项**。

用户可以运行以下命令进行视频编码。

```shell
cd $MMACTION2/tools/data/jester/
bash encode_videos.sh
```

## 步骤 5. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
cd $MMACTION2/tools/data/jester/
bash generate_{rawframes, videos}_filelist.sh
```

## 步骤 6. 检查文件夹结构

在完成所有 Jester 数据集准备流程后，
用户可以获得对应的 RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Jester 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── jester
│   │   ├── jester_{train,val}_list_rawframes.txt
│   │   ├── jester_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── 00001.jpg
│   |   |   |   ├── 00002.jpg
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

关于对 jester 进行训练和验证，可以参考 [基础教程](/docs_zh_CN/getting_started.md)。
