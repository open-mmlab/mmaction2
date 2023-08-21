# 准备数据集

MMAction2 支持许多现有的数据集。在本章中，我们将引导您准备 MMAction2 的数据集。

- [准备数据集](#准备数据集)
  - [关于视频数据格式的说明](#关于视频数据格式的说明)
  - [使用内置数据集](#使用内置数据集)
  - [使用自定义数据集](#使用自定义数据集)
    - [动作识别](#动作识别)
    - [基于骨骼的动作识别](#基于骨骼的动作识别)
    - [基于音频的动作识别](#基于音频的动作识别)
    - [时空动作检测](#时空动作检测)
    - [时序动作定位](#时序动作定位)
  - [使用混合数据集进行训练](#使用混合数据集进行训练)
    - [重复数据集](#重复数据集)
  - [浏览数据集](#浏览数据集)

## 关于视频数据格式的说明

MMAction2 支持两种类型的数据格式：原始帧和视频。前者在之前的项目（如 [TSN](https://github.com/yjxiong/temporal-segment-networks)）中被广泛使用。当 SSD 可用时，这种方法运行速度很快，但无法满足日益增长的数据集需求（例如，最新的 [Kinetics](https://www.deepmind.com/open-source/kinetics) 数据集有 65 万个视频，总帧数将占用几 TB 的空间）。后者可以节省空间，但必须在执行时进行计算密集型的视频解码。为了加快视频解码速度，我们支持几种高效的视频加载库，如 [decord](https://github.com/zhreshold/decord)、[PyAV](https://github.com/PyAV-Org/PyAV) 等。

## 使用内置数据集

MMAction2 已经支持许多数据集，我们在路径 `$MMACTION2/tools/data/` 下提供了用于数据准备的 shell 脚本，请参考[支持的数据集](https://mmaction2.readthedocs.io/zh_CN/latest/datasetzoo_statistics.html)以获取准备特定数据集的详细信息。

## 使用自定义数据集

最简单的方法是将您的数据集转换为现有的数据集格式：

- `RawFrameDataset` 和 `VideoDataset` 用于[动作识别](#动作识别)
- `PoseDataset` 用于[基于骨骼的动作识别](#基于骨骼的动作识别)
- `AudioDataset` 用于[基于音频动作识别](#基于音频动作识别)
- `AVADataset` 用于[时空动作检测](#时空动作检测)
- `ActivityNetDataset` 用于[时序动作定位](#时序动作定位)

在数据预处理之后，用户需要进一步修改配置文件以使用数据集。以下是在原始帧格式中使用自定义数据集的示例。

在 `configs/task/method/my_custom_config.py` 中：

```python
...
# 数据集设置
dataset_type = 'RawframeDataset'
data_root = 'path/to/your/root'
data_root_val = 'path/to/your/root_val'
ann_file_train = 'data/custom/custom_train_list.txt'
ann_file_val = 'data/custom/custom_val_list.txt'
ann_file_test = 'data/custom/custom_val_list.txt'
...
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        ...),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ...),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        ...))
...
```

### 动作识别

动作识别有两种类型的注释文件。

- `RawFrameDataset` 的原始帧注释

  原始帧数据集的注释是一个包含多行的文本文件，每一行表示一个视频的 `frame_directory`（相对路径）、视频的 `total_frames` 和视频的 `label`，用空格分隔。

  以下是一个示例。

  ```
  some/directory-1 163 1
  some/directory-2 122 1
  some/directory-3 258 2
  some/directory-4 234 2
  some/directory-5 295 3
  some/directory-6 121 3
  ```

- `VideoDataset` 的视频注释

  视频数据集的注释是一个包含多行的文本文件，每一行表示一个样本视频，包括 `filepath`（相对路径）和 `label`，用空格分隔。

  以下是一个示例。

  ```
  some/path/000.mp4 1
  some/path/001.mp4 1
  some/path/002.mp4 2
  some/path/003.mp4 2
  some/path/004.mp4 3
  some/path/005.mp4 3
  ```

### 基于骨骼点的动作识别

该任务基于骨骼序列（关键点的时间序列）识别动作类别。我们提供了一些方法来构建自定义的骨骼数据集。

- 从 RGB 视频数据构建

  您需要从视频中提取关键点数据，并将其转换为支持的格式。我们提供了一个[教程](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d/custom_dataset_training.md)，详细介绍了如何执行。

- 从现有关键点数据构建

  假设您已经有了 coco 格式的关键点数据，您可以将它们收集到一个 pickle 文件中。

  每个 pickle 文件对应一个动作识别数据集。pickle 文件的内容是一个字典，包含两个字段：`split` 和 `annotations`

  1. Split：`split` 字段的值是一个字典：键是拆分名称，值是属于特定剪辑的视频标识符列表。
  2. Annotations：`annotations` 字段的值是一个骨骼注释列表，每个骨骼注释是一个字典，包含以下字段：
     - `frame_dir`（str）：对应视频的标识符。
     - `total_frames`（int）：此视频中的帧数。
     - `img_shape`（tuple\[int\]）：视频帧的形状，一个包含两个元素的元组，格式为 `(height, width)`。仅对 2D 骨骼需要。
     - `original_shape`（tuple\[int\]）：与 `img_shape` 相同。
     - `label`（int）：动作标签。
     - `keypoint`（np.ndarray，形状为 `[M x T x V x C]`）：关键点注释。
       - M：人数；
       - T：帧数（与 `total_frames` 相同）；
       - V：关键点数量（NTURGB+D 3D 骨骼为 25，Coco 为 17，OpenPose 为 18 等）；
       - C：关键点坐标的维数（2D 关键点为 C=2，3D 关键点为 C=3）。
     - `keypoint_score`（np.ndarray，形状为 `[M x T x V]`）：关键点的置信度分数。仅对 2D 骨骼需要。

  以下是一个示例：

  ```
  {
      "split":
          {
              'xsub_train':
                  ['S001C001P001R001A001', ...],
              'xsub_val':
                  ['S001C001P003R001A001', ...],
              ...
          }

      "annotations:
          [
              {
                  {
                      'frame_dir': 'S001C001P001R001A001',
                      'label': 0,
                      'img_shape': (1080, 1920),
                      'original_shape': (1080, 1920),
                      'total_frames': 103,
                      'keypoint': array([[[[1032. ,  334.8], ...]]])
                      'keypoint_score': array([[[0.934 , 0.9766, ...]]])
                  },
                  {
                      'frame_dir': 'S001C001P003R001A001',
                      ...
                  },
                  ...

              }
          ]
  }
  ```

  支持其他关键点格式需要进行进一步修改，请参考[自定义数据集](../advanced_guides/customize_dataset.md)。

### 基于音频的动作识别

MMAction2 支持基于 `AudioDataset` 的音频动作识别任务。该任务使用梅尔频谱特征作为输入, 注释文件格式示例如下：

```
ihWykL5mYRI.npy 300 153
lumzQD42AN8.npy 240 321
sWFRmD9Of4s.npy 250 250
w_IpfgRsBVA.npy 300 356
```

每一行代表一个训练样本，以第一行为例，`ihWykL5mYRI.npy` 为梅尔频谱特征的文件名，`300` 为该梅尔频谱特征文件对应的原视频文件的总帧数，`153` 为类别标签。我们分以下两阶段生成所需要的梅尔频谱特征文件数据：

首先，通过视频文件提取`音频文件`:

```
cd $MMACTION2
python tools/data/extract_audio.py ${ROOT} ${DST_ROOT} [--ext ${EXT}] [--num-workers ${N_WORKERS}] \
    [--level ${LEVEL}]
```

- `ROOT`: 视频的根目录。
- `DST_ROOT`: 存放生成音频的根目录。
- `EXT`: 视频的后缀名，如 `mp4`。
- `N_WORKERS`: 使用的进程数量。

下一步，从音频文件生成`梅尔频谱特征`:

```
cd $MMACTION2
python tools/data/build_audio_features.py ${AUDIO_HOME_PATH} ${SPECTROGRAM_SAVE_PATH} [--level ${LEVEL}] \
    [--ext $EXT] [--num-workers $N_WORKERS] [--part $PART]
```

- `AUDIO_HOME_PATH`: 音频文件的根目录。
- `SPECTROGRAM_SAVE_PATH`: 存放生成音频特征的根目录。
- `EXT`: 音频的后缀名，如 `m4a`。
- `N_WORKERS`: 使用的进程数量。
- `PART`: 将完整的解码任务分为几部分并执行其中一份。如 `2/5` 表示将所有待解码数据分成 5 份，并对其中的第 2 份进行解码。这一选项在用户有多台机器时发挥作用。

### 时空动作检测

MMAction2 支持基于 `AVADataset` 的时空动作检测任务。注释包含真实边界框和提议边界框。

- 真实边界框
  真实边界框是一个包含多行的 csv 文件，每一行是一个帧的检测样本，格式如下：

  video_identifier, time_stamp, lt_x, lt_y, rb_x, rb_y, label, entity_id
  每个字段的含义如下：
  `video_identifier`：对应视频的标识符
  `time_stamp`：当前帧的时间戳
  `lt_x`：左上角点的规范化 x 坐标
  `lt_y`：左上角点的规范化 y 坐标
  `rb_y`：右下角点的规范化 x 坐标
  `rb_y`：右下角点的规范化 y 坐标
  `label`：动作标签
  `entity_id`：一个唯一的整数，允许将此框与该视频相邻帧中描绘同一个人的其他框连接起来

  以下是一个示例：

  ```
  _-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,12,0
  _-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,74,0
  ...
  ```

- 提议边界框
  提议边界框是由一个人体检测器生成的 pickle 文件，通常需要在目标数据集上进行微调。pickle 文件包含一个带有以下数据结构的字典：

  `{'video_identifier,time_stamp': bbox_info}`

  video_identifier（str）：对应视频的标识符
  time_stamp（int）：当前帧的时间戳
  bbox_info（np.ndarray，形状为`[n, 5]`）：检测到的边界框，\<x1> \<y1> \<x2> \<y2> \<score>。x1、x2、y1、y2 是相对于帧大小归一化的值，范围为 0.0-1.0。

### 时序动作定位

我们支持基于 `ActivityNetDataset` 的时序动作定位。ActivityNet 数据集的注释是一个 json 文件。每个键是一个视频名，相应的值是视频的元数据和注释。

以下是一个示例：

```
{
  "video1": {
      "duration_second": 211.53,
      "duration_frame": 6337,
      "annotations": [
          {
              "segment": [
                  30.025882995319815,
                  205.2318595943838
              ],
              "label": "Rock climbing"
          }
      ],
      "feature_frame": 6336,
      "fps": 30.0,
      "rfps": 29.9579255898
  },
  "video2": {...
  }
  ...
}
```

## 使用混合数据集进行训练

MMAction2 还支持混合数据集进行训练。目前，它支持重复数据集。

### 重复数据集

我们使用 `RepeatDataset` 作为包装器来重复数据集。例如，假设原始数据集为 `Dataset_A`，要重复它，配置如下所示

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # 这是 Dataset_A 的原始配置
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

## 浏览数据集

即将推出...
