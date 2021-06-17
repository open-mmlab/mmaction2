# 准备数据

本文为 MMAction2 的数据准备提供一些指南。

<!-- TOC -->

- [视频格式数据的一些注意事项](#视频格式数据的一些注意事项)
- [获取数据](#获取数据)
  - [准备视频](#准备视频)
  - [提取帧](#提取帧)
    - [denseflow 的替代项](#denseflow-的替代项)
  - [生成文件列表](#生成文件列表)
  - [准备音频](#准备音频)

<!-- TOC -->

## 视频格式数据的一些注意事项

MMAction2 支持两种数据类型：原始帧和视频。前者在过去的项目中经常出现，如 TSN。
如果能把原始帧存储在固态硬盘上，处理帧格式的数据是非常快的，但对于大规模的数据集来说，原始帧需要占据大量的磁盘空间。
（举例来说，最新版本的 [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) 有 650K 个视频，其所有原始帧需要占据几个 TB 的磁盘空间。）
视频格式的数据能够节省很多空间，但在运行模型时，必须进行视频解码，算力开销很大。
为了加速视频解码，MMAction2 支持了若干种高效的视频加载库，如 [decord](https://github.com/zhreshold/decord), [PyAV](https://github.com/PyAV-Org/PyAV) 等。

## 获取数据

本文介绍如何构建自定义数据集。
与上述数据集相似，推荐用户把数据放在 `$MMACTION2/data/$DATASET` 中。

### 准备视频

请参照官网或官方脚本准备视频。
注意，应该按照下面两种方法之一来组织视频数据文件夹结构：

(1) 形如 `${CLASS_NAME}/${VIDEO_ID}` 的两级文件目录结构，这种结构推荐在动作识别数据集中使用（如 UCF101 和 Kinetics）

(2) 单级文件目录结构，这种结构推荐在动作检测数据集或者多标签数据集中使用（如 THUMOS14）

### 提取帧

若想同时提取帧和光流，可以使用 OpenMMLab 准备的 [denseflow](https://github.com/open-mmlab/denseflow) 工具。
因为不同的帧提取工具可能产生不同数量的帧，建议使用同一工具来提取 RGB 帧和光流，以避免它们的数量不同。

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
- `--use-opencv`: 是否使用 OpenCV 提取 RGB 帧
- `--mixed-ext`: 说明是否处理不同文件类型的视频文件

根据实际经验，推荐设置为：

1. 将 `$OUT_FOLDER` 设置为固态硬盘上的文件夹。
2. 软连接 `$OUT_FOLDER` 到 `$MMACTION2/data/$DATASET/rawframes`
3. 使用 `new-short` 而不是 `new-width` 和 `new-height` 来调整图像尺寸

```shell
ln -s ${YOUR_FOLDER} $MMACTION2/data/$DATASET/rawframes
```

#### denseflow 的替代项

如果用户因依赖要求（如 Nvidia 显卡驱动版本），无法安装 [denseflow](https://github.com/open-mmlab/denseflow)，
或者只需要一些关于光流提取的快速演示，可用 Python 脚本 `tools/misc/flow_extraction.py` 替代 denseflow。
这个脚本可用于一个或多个视频提取 RGB 帧和光流。注意，由于该脚本时在 CPU 上运行光流算法，其速度比 denseflow 慢很多。

```shell
python tools/misc/flow_extraction.py --input ${INPUT} [--prefix ${PREFIX}] [--dest ${DEST}] [--rgb-tmpl ${RGB_TMPL}] \
    [--flow-tmpl ${FLOW_TMPL}] [--start-idx ${START_IDX}] [--method ${METHOD}] [--bound ${BOUND}] [--save-rgb]
```

- `INPUT`: 用于提取帧的视频，可以是单个视频或一个视频列表，视频列表应该是一个 txt 文件，并且只包含视频文件名，不包含目录
- `PREFIX`: 输入视频的前缀，当输入是一个视频列表时使用
- `DEST`: 保存提取出的帧的位置
- `RGB_TMPL`:  RGB 帧的文件名格式
- `FLOW_TMPL`: 光流的文件名格式
- `START_IDX`: 提取帧的开始索引
- `METHOD`: 用于生成光流的方法
- `BOUND`: 光流的最大值
- `SAVE_RGB`: 同时保存提取的 RGB 帧

### 生成文件列表

MMAction2 提供了便利的脚本用于生成文件列表。在完成视频下载（或更进一步完成视频抽帧）后，用户可以使用如下的脚本生成文件列表。

```shell
cd $MMACTION2
python tools/data/build_file_list.py ${DATASET} ${SRC_FOLDER} [--rgb-prefix ${RGB_PREFIX}] \
    [--flow-x-prefix ${FLOW_X_PREFIX}] [--flow-y-prefix ${FLOW_Y_PREFIX}] [--num-split ${NUM_SPLIT}] \
    [--subset ${SUBSET}] [--level ${LEVEL}] [--format ${FORMAT}] [--out-root-path ${OUT_ROOT_PATH}] \
    [--seed ${SEED}] [--shuffle]
```

- `DATASET`: 所要准备的数据集，例如：`ucf101` , `kinetics400` , `thumos14` , `sthv1` , `sthv2` 等。
- `SRC_FOLDER`: 存放对应格式的数据的目录：
  - 如目录为 "$MMACTION2/data/$DATASET/rawframes"，则需设置 `--format rawframes`。
  - 如目录为 "$MMACTION2/data/$DATASET/videos"，则需设置 `--format videos`。
- `RGB_PREFIX`: RGB 帧的文件前缀。
- `FLOW_X_PREFIX`: 光流 x 分量帧的文件前缀。
- `FLOW_Y_PREFIX`: 光流 y 分量帧的文件前缀。
- `NUM_SPLIT`: 数据集总共的划分个数。
- `SUBSET`: 需要生成文件列表的子集名称。可选项为 `train`, `val`, `test`。
- `LEVEL`: 目录级别数量，1 表示一级目录（数据集中所有视频或帧文件夹位于同一目录）， 2 表示二级目录（数据集中所有视频或帧文件夹按类别存放于各子目录）。
- `FORMAT`: 需要生成文件列表的源数据格式。可选项为 `rawframes`, `videos`。
- `OUT_ROOT_PATH`: 生成文件的根目录。
- `SEED`: 随机种子。
- `--shuffle`: 是否打乱生成的文件列表。

至此为止，用户可参考 [基础教程](getting_started.md) 来进行模型的训练及测试。

### 准备音频

MMAction2 还提供如下脚本来提取音频的波形并生成梅尔频谱。

```shell
cd $MMACTION2
python tools/data/extract_audio.py ${ROOT} ${DST_ROOT} [--ext ${EXT}] [--num-workers ${N_WORKERS}] \
    [--level ${LEVEL}]
```

- `ROOT`: 视频的根目录。
- `DST_ROOT`: 存放生成音频的根目录。
- `EXT`: 视频的后缀名，如 `.mp4`。
- `N_WORKERS`: 使用的进程数量。

成功提取出音频后，用户可参照 [配置文件](/configs/audio_recognition/tsn_r50_64x1x1_kinetics400_audio.py) 在线解码并生成梅尔频谱。如果音频文件的目录结构与帧文件夹一致，用户可以直接使用帧数据所用的标注文件作为音频数据的标注文件。在线解码的缺陷在于速度较慢，因此，MMAction2 也提供如下脚本用于离线地生成梅尔频谱。

```shell
cd $MMACTION2
python tools/data/build_audio_features.py ${AUDIO_HOME_PATH} ${SPECTROGRAM_SAVE_PATH} [--level ${LEVEL}] \
    [--ext $EXT] [--num-workers $N_WORKERS] [--part $PART]
```

- `AUDIO_HOME_PATH`: 音频文件的根目录。
- `SPECTROGRAM_SAVE_PATH`: 存放生成音频特征的根目录。
- `EXT`: 音频的后缀名，如 `.m4a`。
- `N_WORKERS`: 使用的进程数量。
- `PART`: 将完整的解码任务分为几部分并执行其中一份。如 `2/5` 表示将所有待解码数据分成 5 份，并对其中的第 2 份进行解码。这一选项在用户有多台机器时发挥作用。

梅尔频谱特征所对应的标注文件与帧文件夹一致，用户可以直接复制 `dataset_[train/val]_list_rawframes.txt` 并将其重命名为 `dataset_[train/val]_list_audio_feature.txt`。
