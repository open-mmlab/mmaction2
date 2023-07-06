# 自定义数据集

在本教程中，我们将介绍如何通过在线转换来自定义你的数据集。

- [自定义数据集](#自定义数据集)
  - [MMAction2 数据集概述](#mmaction2-数据集概述)
  - [定制新的数据集](#定制新的数据集)
  - [为 PoseDataset 自定义关键点格式](#为-posedataset-自定义关键点格式)

## MMAction2 数据集概述

MMAction2 提供了任务特定的 `Dataset` 类，例如用于动作识别的 `VideoDataset`/`RawframeDataset`，用于时空动作检测的 `AVADataset`，用于基于骨骼的动作识别的`PoseDataset`。这些任务特定的数据集只需要实现 `load_data_list(self)` 来从注释文件生成数据列表。剩下的函数由超类（即 `BaseActionDataset` 和 `BaseDataset`）自动处理。下表显示了模块的继承关系和主要方法。

| 类名                           | 类方法                                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MMAction2::VideoDataset`      | `load_data_list(self)` <br> 从注释文件中构建数据列表。                                                                                                        |
| `MMAction2::BaseActionDataset` | `get_data_info(self, idx)` <br> 给定 `idx`，从数据列表中返回相应的数据样本。                                                                                  |
| `MMEngine::BaseDataset`        | `__getitem__(self, idx)` <br> 给定 `idx`，调用 `get_data_info` 获取数据样本，然后调用 `pipeline` 在 `train_pipeline` 或 `val_pipeline` 中执行数据变换和增强。 |

## 定制新的数据集类

大多数情况下，把你的数据集离线转换成指定格式是首选方法，但 MMAction2 提供了一个方便的过程来创建一个定制的 `Dataset` 类。如前所述，任务特定的数据集只需要实现 `load_data_list(self)` 来从注释文件生成数据列表。请注意，`data_list` 中的元素是包含后续流程中必要字段的 `dict`。

以 `VideoDataset` 为例，`train_pipeline`/`val_pipeline` 在 `DecordInit` 中需要 `'filename'`，在 `PackActionInputs` 中需要 `'label'`。因此，`data_list` 中的数据样本必须包含2个字段：`'filename'`和`'label'`。
请参考[定制数据流水线](customize_pipeline.md)以获取有关 `pipeline` 的更多详细信息。

```
data_list.append(dict(filename=filename, label=label))
```

`AVADataset` 会更加复杂，`data_list` 中的数据样本包含有关视频数据的几个字段。此外，它重写了 `get_data_info(self, idx)` 以转换在时空动作检测数据流水线中需要用的字段。

```python

class AVADataset(BaseActionDataset):
  ...

   def load_data_list(self) -> List[dict]:
      ...
        video_info = dict(
            frame_dir=frame_dir,
            video_id=video_id,
            timestamp=int(timestamp),
            img_key=img_key,
            shot_info=shot_info,
            fps=self._FPS,
            ann=ann)
            data_list.append(video_info)
        data_list.append(video_info)
      return data_list

  def get_data_info(self, idx: int) -> dict:
      ...
      ann = data_info.pop('ann')
      data_info['gt_bboxes'] = ann['gt_bboxes']
      data_info['gt_labels'] = ann['gt_labels']
      data_info['entity_ids'] = ann['entity_ids']
      return data_info
```

## 为 PoseDataset 自定义关键点格式

MMAction2 目前支持三种关键点格式：`coco`，`nturgb+d` 和 `openpose`。如果你使用其中一种格式，你可以简单地在以下模块中指定相应的格式：

对于图卷积网络，如 AAGCN，STGCN，...

- `pipeline`：在 `JointToBone` 中的参数 `dataset`。
- `backbone`：在图卷积网络中的参数 `graph_cfg`。

对于 PoseC3D：

- `pipeline`：在 `Flip` 中，根据关键点的对称关系指定 `left_kp` 和 `right_kp`。
- `pipeline`：在 `GeneratePoseTarget` 中，如果 `with_limb` 为 `True`，指定`skeletons`，`left_limb`，`right_limb`，如果 `with_kp` 为 `True`，指定`left_kp` 和 `right_kp`。

如果使用自定义关键点格式，需要在 `backbone` 和 `pipeline` 中都包含一个新的图布局。这个布局将定义关键点及其连接关系。

以 `coco` 数据集为例，我们在 `Graph` 中定义了一个名为 `coco` 的布局。这个布局的 `inward` 连接包括所有节点连接，每个**向心**连接由一个节点元组组成。`coco`的额外设置包括将节点数指定为 `17`，将 `node 0` 设为中心节点。

```python

self.num_node = 17
self.inward = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
self.center = 0
```

同样，我们在 `JointToBone` 中定义了 `pairs`，添加了一个 bone `(0, 0)` 以使 bone 的数量对齐到 joint。coco数据集的 `pairs` 如下所示，`JointToBone` 中的 `pairs` 的顺序无关紧要。

```python

self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2),
                (5, 0), (6, 0), (7, 5), (8, 6), (9, 7),
                (10, 8), (11, 0), (12, 0), (13, 11), (14, 12),
                (15, 13), (16, 14))
```

要使用你的自定义关键点格式，只需定义上述设置为你的图结构，并在你的配置文件中指定它们，如下所示。在这个例子中，我们将使用 `STGCN`，其中 `n` 表示类别的数量，`custom_dataset` 在 `Graph` 和 `JointToBone` 中定义。

```python
model = dict(
  type='RecognizerGCN',
  backbone=dict(
      type='STGCN', graph_cfg=dict(layout='custom_dataset', mode='stgcn_spatial')),
  cls_head=dict(type='GCNHead', num_classes=n, in_channels=256))

train_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

val_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

test_pipeline = [
  ...
  dict(type='GenSkeFeat', dataset='custom_dataset'),
  ...]

```

只需简单地指定自定义布局，你就可以使用你自己的关键点格式进行训练和测试了。通过这种方式，MMAction2 为用户提供了很大的灵活性，允许用户自定义他们的数据集和关键点格式，以满足他们特定的需求。

以上就是关于如何自定义你的数据集的一些方法。希望这个教程能帮助你理解MMAction2的数据集结构，并教给你如何根据自己的需求创建新的数据集。虽然这可能需要一些编程知识，但是 MMAction2 试图使这个过程尽可能简单。通过了解这些基本概念，你将能够更好地控制你的数据，从而改进你的模型性能。
