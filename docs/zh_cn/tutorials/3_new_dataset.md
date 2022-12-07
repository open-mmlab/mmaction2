# 教程 3：如何增加新数据集

在本教程中，我们将介绍一些有关如何按已支持的数据格式进行数据组织，和组合已有数据集来自定义数据集的方法。

<!-- TOC -->

- [教程 3：如何增加新数据集](#教程-3如何增加新数据集)
  - [通过重组数据来自定义数据集](#通过重组数据来自定义数据集)
    - [将数据集重新组织为现有格式](#将数据集重新组织为现有格式)
    - [自定义数据集的示例](#自定义数据集的示例)
  - [通过组合已有数据集来自定义数据集](#通过组合已有数据集来自定义数据集)
    - [重复数据集](#重复数据集)

<!-- TOC -->

## 通过重组数据来自定义数据集

### 将数据集重新组织为现有格式

最简单的方法是将数据集转换为现有的数据集格式（RawframeDataset 或 VideoDataset）。

有三种标注文件：

- 帧标注（rawframe annotation）

  帧数据集（rawframe dataset）标注文件由多行文本组成，每行代表一个样本，每个样本分为三个部分，分别是 `帧（相对）文件夹`（rawframe directory of relative path），
  `总帧数`（total frames）以及 `标签`（label），通过空格进行划分

  示例如下：

  ```
  some/directory-1 163 1
  some/directory-2 122 1
  some/directory-3 258 2
  some/directory-4 234 2
  some/directory-5 295 3
  some/directory-6 121 3
  ```

- 视频标注（video annotation）

  视频数据集（video dataset）标注文件由多行文本组成，每行代表一个样本，每个样本分为两个部分，分别是 `文件（相对）路径`（filepath of relative path）
  和 `标签`（label），通过空格进行划分

  示例如下：

  ```
  some/path/000.mp4 1
  some/path/001.mp4 1
  some/path/002.mp4 2
  some/path/003.mp4 2
  some/path/004.mp4 3
  some/path/005.mp4 3
  ```

- ActivityNet 标注

  ActivityNet 数据集的标注文件是一个 json 文件。每个键是一个视频名，其对应的值是这个视频的元数据和注释。

  示例如下：

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
    "video2": {
        "duration_second": 26.75,
        "duration_frame": 647,
        "annotations": [
            {
                "segment": [
                    2.578755070202808,
                    24.914101404056165
                ],
                "label": "Drinking beer"
            }
        ],
        "feature_frame": 624,
        "fps": 24.0,
        "rfps": 24.1869158879
    }
  }
  ```

有两种使用自定义数据集的方法：

- 在线转换

  用户可以通过继承 [BaseDataset](/mmaction/datasets/base.py) 基类编写一个新的数据集类，并重写三个抽象类方法：
  `load_annotations(self)`，`evaluate(self, results, metrics, logger)` 和 `dump_results(self, results, out)`，
  如 [RawframeDataset](/mmaction/datasets/rawframe_dataset.py)，[VideoDataset](/mmaction/datasets/video_dataset.py) 或 [ActivityNetDataset](/mmaction/datasets/activitynet_dataset.py)。

- 本地转换

  用户可以转换标注文件格式为上述期望的格式，并将其存储为 pickle 或 json 文件，然后便可以应用于 `RawframeDataset`，`VideoDataset` 或 `ActivityNetDataset` 中。

数据预处理后，用户需要进一步修改配置文件以使用数据集。 这里展示了以帧形式使用自定义数据集的例子：

在 `configs/task/method/my_custom_config.py` 下：

```python
...
# 数据集设定
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

### 自定义数据集的示例

假设注释在文本文件中以新格式显示，并且图像文件名具有类似 “img_00005.jpg” 的模板。
那么视频注释将以以下形式存储在文本文件 `annotation.txt` 中。

```
#文件夹,总帧数,类别
D32_1gwq35E,299,66
-G-5CJ0JkKY,249,254
T4h1bvOd9DA,299,33
4uZ27ivBl00,299,341
0LfESFkfBSw,249,186
-YIsNpBEx6c,299,169
```

在 `mmaction/datasets/my_dataset.py` 中创建新数据集加载数据

```python
import copy
import os.path as osp

import mmcv

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class MyDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg'):
        super(MyDataset, self).__init__(ann_file, pipeline, test_mode)

        self.filename_tmpl = filename_tmpl

    def load_annotations(self):
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                if line.startswith("directory"):
                    continue
                frame_dir, total_frames, label = line.split(',')
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_infos.append(
                    dict(
                        frame_dir=frame_dir,
                        total_frames=int(total_frames),
                        label=int(label)))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        return self.pipeline(results)

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 topk=(1, 5),
                 logger=None):
        pass
```

然后在配置文件中，用户可通过如下修改来使用 `MyDataset`：

```python
dataset_A_train = dict(
    type='MyDataset',
    ann_file=ann_file_train,
    pipeline=train_pipeline
)
```

## 通过组合已有数据集来自定义数据集

MMAction2 还支持组合已有数据集以进行训练。 目前，它支持重复数据集（repeat dataset）。

### 重复数据集

MMAction2 使用 “RepeatDataset” 作为包装器来重复数据集。例如，假设原始数据集为 “Dataset_A”，
为了重复此数据集，可设置配置如下：

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
