# 准备数据

本文为 MMAction2 的数据准备提供一些指南。

<!-- TOC -->

- [视频格式数据的一些注意事项](#视频格式数据的一些注意事项)
- [获取数据](#获取数据)
  - [准备视频](#准备视频)
  - [提取帧](#提取帧)
    - [denseflow 的替代项](#denseflow-的替代项)
  - [Generate file list](#generate-file-list)
  - [Prepare audio](#prepare-audio)

<!-- TOC -->

## 视频格式数据的一些注意事项

MMAction2 支持两种数据类型：原始帧和视频。前者在过去的项目中，如 TSN，经常出现。
如果能把原始帧存储在固态硬盘上，处理帧格式的数据是非常快的，但帧格式难以拓展到快速扩充的数据集上。
（举例来说，最新版本的 [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) 有 650K 个视频，其所有原始帧需要占据几个 TB 的磁盘空间。）
视频格式的数据能够节省很多空间，但在运行模型时，必须做算力开销很大的视频解码。
为了加速视频解码，MMAction2 支持了若干种高效的视频加载库，如 [decord](https://github.com/zhreshold/decord), [PyAV](https://github.com/PyAV-Org/PyAV) 等等。

## 获取数据

下面的指南能为用户在自定义数据集上实验，提供一些帮助。
与上述数据集相似，推荐用户把数据放在 `$MMACTION2/data/$DATASET` 中。

### 准备视频

请参照官网或官方脚本准备视频。
注意，应该按照下面两种方法之一来组织视频数据文件夹结构：

(1). 形如 `${CLASS_NAME}/${VIDEO_ID}` 的两级文件目录结构，这种结构推荐在动作识别数据集中使用（如 UCF101 和 Kinetics）

(2). 单级文件目录结构，这种结构推荐在动作检测数据集或者多标签数据集中使用（如 THUMOS14）

### 提取帧

若想同时提取帧和光流，可以使用 MMAction2 准备的 [denseflow](https://github.com/open-mmlab/denseflow) 工具。
因为不同的帧提取工具可能产生不同数量的帧，最好使用同一工具来提取帧和光流，以避免帧和光流提取数量的不匹配。

```shell
python build_rawframes.py ${SRC_FOLDER} ${OUT_FOLDER} [--task ${TASK}] [--level ${LEVEL}] \
    [--num-worker ${NUM_WORKER}] [--flow-type ${FLOW_TYPE}] [--out-format ${OUT_FORMAT}] \
    [--ext ${EXT}] [--new-width ${NEW_WIDTH}] [--new-height ${NEW_HEIGHT}] [--new-short ${NEW_SHORT}] \
    [--resume] [--use-opencv] [--mixed-ext]
```

- `SRC_FOLDER`: 视频源文件夹
- `OUT_FOLDER`: 存储提取出的帧和光流的根文件夹
- `TASK`: 提取任务，说明提取帧，光流，还是都提取，选项为 `rgb`, `flow`, `both`
- `LEVEL`: 目录层级。1 指单级文件目录，2 指两级文件目录
- `NUM_WORKER`: 提取原始帧的线程数
- `FLOW_TYPE`: 提取的光流类型，如 `None`, `tvl1`, `warp_tvl1`, `farn`, `brox`
- `OUT_FORMAT`: 提取帧的输出文件类型，如 `jpg`, `h5`, `png`
- `EXT`: 视频文件后缀名，如 `avi`, `mp4`
- `NEW_WIDTH`: 调整尺寸后，输出图像的宽
- `NEW_HEIGHT`: 调整尺寸后，输出图像的高
- `NEW_SHORT`: 等比例缩放图片后，输出图像的短边长
- `--resume`: 是否接续之前的光流提取任务，还是覆盖之前的输出结果重新提取
- `--use-opencv`: 是否使用 OpenCV 提取图像帧
- `--mixed-ext`: 说明是否处理不同文件类型的视频文件

一些根据实际经验得来的推荐设置是

1. 将 `$OUT_FOLDER` 设置为固态硬盘上的文件夹。
2. 软连接 `$OUT_FOLDER` 到 `$MMACTION2/data/$DATASET/rawframes`
3. 使用 `new-short` 而不是 `new-width` 和 `new-height` 来调整图像尺寸

```shell
ln -s ${YOUR_FOLDER} $MMACTION2/data/$DATASET/rawframes
```

#### denseflow 的替代项

如果用户的设备不能满足安装 [denseflow](https://github.com/open-mmlab/denseflow) 的依赖要求（比如 Nvidia 显卡驱动版本），
或者只想看一些关于光流提取的快速演示，MMAction2 提供了一个 Python 脚本 `tools/flow_extraction.py` 作为 denseflow 的替代。
这个脚本能用来为一个或几个视频提取图像帧和光流。注意这个脚本比 denseflow 慢很多，因为它在 CPU 上运行光流算法。

```shell
python tools/flow_extraction.py --input ${INPUT} [--prefix ${PREFIX}] [--dest ${DEST}] [--rgb-tmpl ${RGB_TMPL}] \
    [--flow-tmpl ${FLOW_TMPL}] [--start-idx ${START_IDX}] [--method ${METHOD}] [--bound ${BOUND}] [--save-rgb]
```

- `INPUT`: 用于提取帧的视频，可以是单个视频或一个视频列表，视频列表应该是一个 txt 文件，并且只包含视频文件名，不包含目录
- `PREFIX`: 输入视频的前缀，当输入是一个视频列表时使用
- `DEST`: 保存提取出的帧的位置
- `RGB_TMPL`: 图像帧的文件名格式
- `FLOW_TMPL`: 光流的文件名格式
- `START_IDX`: 提取帧的开始索引
- `METHOD`: 用于生成光流的方法
- `BOUND`: 光流的最大值
- `SAVE_RGB`: 同时保存提取的图像帧
