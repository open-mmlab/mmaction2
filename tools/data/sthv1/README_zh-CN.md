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
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/sthv1/`。

## 步骤 1. 下载标注文件

由于 Something-Something V1 的官方网站已经失效，用户需要通过第三方源下载原始数据集。下载好的标注文件需要放在 `$MMACTION2/data/sthv1/annotations` 文件夹下。

## 步骤 2. 准备 RGB 帧

官方数据集并未提供原始视频文件，只提供了对原视频文件进行抽取得到的 RGB 帧，用户可在第三方源直接下载视频帧。

将下载好的压缩文件放在 `$MMACTION2/data/sthv1/` 文件夹下，并使用以下脚本进行解压。

```shell
cd $MMACTION2/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION2/tools/data/sthv1/
```

如果用户只想使用 RGB 帧，则可以跳过中间步骤至步骤 5 以直接生成视频帧的文件列表。
由于官网的 JPG 文件名形如 "%05d.jpg" （比如，"00001.jpg"），需要在配置文件的 `data.train`, `data.val` 和 `data.test` 处添加 `"filename_tmpl='{:05}.jpg'"` 代码，以修改文件名模板。

```
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
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

如果用户只想使用原 RGB 帧加载训练，则该部分是 **可选项**。

在抽取视频帧和光流之前，请参考 [安装指南](/docs/zh_cn/get_started/installation.md) 安装 [denseflow](https://github.com/open-mmlab/denseflow)。

如果拥有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 中。

可以运行以下命令为 SSD 建立软链接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/sthv1_extracted/
ln -s /mnt/SSD/sthv1_extracted/ ../../../data/sthv1/rawframes
```

如果想抽取光流，则可以运行以下脚本从 RGB 帧中抽取出光流。

```shell
cd $MMACTION2/tools/data/sthv1/
bash extract_flow.sh
```

## 步骤 4: 编码视频

如果用户只想使用 RGB 帧加载训练，则该部分是 **可选项**。

用户可以运行以下命令进行视频编码。

```shell
cd $MMACTION2/tools/data/sthv1/
bash encode_videos.sh
```

## 步骤 5. 生成文件列表

用户可以通过运行以下命令生成帧和视频格式的文件列表。

```shell
cd $MMACTION2/tools/data/sthv1/
bash generate_{rawframes, videos}_filelist.sh
```

## 步骤 6. 检查文件夹结构

在完成所有 Something-Something V1 数据集准备流程后，
用户可以获得对应的 RGB + 光流文件，视频文件以及标注文件。

在整个 MMAction2 文件夹下，Something-Something V1 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
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

关于对 Something-Something V1 进行训练和验证，请参考 [训练和测试教程](/docs/en/user_guides/train_test.md)。
