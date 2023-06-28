# 自定义数据流水线

在本教程中，我们将介绍如何为你的任务构建数据流水线（即，数据转换）的一些方法。

- [自定义数据流水线](#自定义数据流水线)
  - [数据流水线设计](#数据流水线设计)
  - [修改训练/测试数据流水线](#修改训练/测试数据流水线)
    - [加载](#加载)
    - [采样帧和其他处理](#采样帧和其他处理)
    - [格式化](#格式化)
  - [添加新的数据转换](#添加新的数据转换)

## 数据流水线设计

数据流水线指的是从数据集索引样本时处理数据样本字典的过程，它包括一系列的数据转换。每个数据转换接受一个 `dict` 作为输入，对其进行处理，并产生一个 `dict` 作为输出，供序列中的后续数据转换使用。

以下是一个例子，用于使用 `VideoDataset` 在 Kinetics 上训练 SlowFast 的数据流水线。这个数据流水线首先使用 [`decord`](https://github.com/dmlc/decord) 读取原始视频并随机采样一个视频剪辑，该剪辑包含 `32` 帧，帧间隔为 `2`。然后，它对所有帧应用随机大小调整的裁剪和随机水平翻转，然后将数据形状格式化为 `NCTHW`，在这个例子中，它是 `(1, 3, 32, 224, 224)`。

```python
train_pipeline = [
    dict(type='DecordInit',),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
```

MMAction2 中所有可用的数据转换的详细列表可以在 [mmaction.datasets.transforms](mmaction.datasets.transforms) 中找到。

## 修改训练/测试数据流水线

MMAction2 的数据流水线非常灵活，因为几乎每一步的数据预处理都可以从配置文件中进行配置。然而，对于一些用户来说，这种多样性可能会让人感到不知所措。

以下是一些用于构建动作识别任务数据流水线的一般实践和指南。

### 加载

在数据流水线的开始，通常是加载视频。然而，如果帧已经被提取出来，你应该使用 `RawFrameDecode` 并修改数据集类型为 `RawframeDataset`。

```python
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
```

如果你需要从具有不同格式（例如，`pkl`，`bin`等）的文件或从特定位置加载数据，你可以创建一个新的加载转换并将其包含在数据流水线的开始。有关更多详细信息，请参阅[添加新的数据转换](#添加新的数据转换)。

### 采样帧和其他处理

在训练和测试过程中，我们可能会有从视频中采样帧的不同策略。

例如，当测试 SlowFast 时，我们会均匀地采样多个剪辑，如下所示：

```python
test_pipeline = [
    ...
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    ...
]
```

在上述例子中，每个视频将均匀地采样10个视频剪辑，每个剪辑包含32帧。 `test_mode=True` 用于实现这一点，与训练期间的随机采样相反。

另一个例子涉及 `TSN/TSM` 模型，它们从视频中采样多个片段：

```python
train_pipeline = [
    ...
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    ...
]
```

通常，数据流水线中的数据增强只处理视频级的转换，例如调整大小或裁剪，而不处理像视频标准化或 mixup/cutmix 这样的转换。这是因为我们可以在批量视频数据上进行视频标准化和 mixup/cutmix，以使用 GPU 加速处理。要配置视频标准化和 mixup/cutmix，请使用 [mmaction.models.utils.data_preprocessor](mmaction.models.utils.data_preprocessor)。

### 格式化

格式化涉及从数据信息字典中收集训练数据，并将其转换为与模型兼容的格式。

在大多数情况下，你可以简单地使用 [`PackActionInputs`](mmaction.datasets.transforms.PackActionInputs)，它将以 `NumPy Array` 格式的图像转换为 `PyTorch Tensor`，并将地面真实类别信息和其他元信息打包为一个类似字典的对象 [`ActionDataSample`](mmaction.structures.ActionDataSample)。

```python
train_pipeline = [
    ...
    dict(type='PackActionInputs'),
]
```

## 添加新的数据转换

1. 要创建一个新的数据转换，编写一个新的转换类在一个 Python 文件中，例如，名为 `my_transforms.py`。数据转换类必须继承 [`mmcv.transforms.BaseTransform`](mmcv.transforms.BaseTransform) 类，并重写 `transform` 方法，该方法接受一个 `dict` 作为输入并返回一个 `dict`。最后，将 `my_transforms.py` 放在 `mmaction/datasets/transforms/` 文件夹中。

   ```python
   from mmcv.transforms import BaseTransform
   from mmaction.datasets import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):
        def __init__(self, msg):
            self.msg = msg

       def transform(self, results):
           # 修改数据信息字典 `results`。
           print(msg, 'MMAction2.')
           return results
   ```

2. 在 `mmaction/datasets/transforms/__init__.py` 中导入新类。

   ```python
   ...
   from .my_transform import MyTransform

   __all__ = [
       ..., 'MyTransform'
   ]
   ```

3. 在配置文件中使用它。

   ```python
   train_pipeline = [
       ...
       dict(type='MyTransform', msg='Hello!'),
       ...
   ]
   ```
