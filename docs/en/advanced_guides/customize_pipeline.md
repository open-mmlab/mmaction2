# Customize Data Pipeline

In this tutorial, we will introduce some methods about how to build the data pipeline (i.e., data transformations) for your tasks.

- [Customize Data Pipeline](#customize-data-pipeline)
  - [Design of Data Pipeline](#design-of-data-pipeline)
  - [Modify the Training/Testing Pipeline](#modify-the-trainingtest-pipeline)
    - [Loading](#loading)
    - [Sampling Frames and Other Processing](#sampling-frames-and-other-processing)
    - [Formatting](#formatting)
  - [Add New Data Transforms](#add-new-data-transforms)

## Design of Data Pipeline

The data pipeline refers to the procedure of handling the data sample dict when indexing a sample from the dataset, and comprises a series of data transforms. Each data transform accepts a `dict` as input, processes it, and produces a `dict` as output for the subsequent data transform in the sequence.

Below is an example data pipeline for training SlowFast on Kinetics using `VideoDataset`. The pipeline initially employs [`decord`](https://github.com/dmlc/decord) to read the raw videos and randomly sample one video clip, which comprises `32` frames with a frame interval of `2`. Subsequently, it applies random resized crop and random horizontal flip to all frames before formatting the data shape as `NCTHW`, which is `(1, 3, 32, 224, 224)` in this example.

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

A comprehensive list of all available data transforms in MMAction2 can be found in the [mmaction.datasets.transforms](mmaction.datasets.transforms).

## Modify the Training/Testing Pipeline

The data pipeline in MMAction2 is highly adaptable, as nearly every step of the data preprocessing can be configured from the config file. However, the wide array of options may be overwhelming for some users.

Below are some general practices and guidance for building a data pipeline for action recognition tasks.

### Loading

At the beginning of a data pipeline, it is customary to load videos. However, if the frames have already been extracted, you should utilize `RawFrameDecode` and modify the dataset type to `RawframeDataset`.

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

If you need to load data from files with distinct formats (e.g., `pkl`, `bin`, etc.) or from specific locations, you may create a new loading transform and include it at the beginning of the data pipeline. Please refer to [Add New Data Transforms](#add-new-data-transforms) for more details.

### Sampling Frames and Other Processing

During training and testing, we may have different strategies to sample frames from the video.

For instance, when testing SlowFast, we uniformly sample multiple clips as follows:

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

In the above example, 10 video clips, each comprising 32 frames, will be uniformly sampled from each video. `test_mode=True` is employed to accomplish this, as opposed to random sampling during training.

Another example involves `TSN/TSM` models, which sample multiple segments from the video:

```python
train_pipeline = [
    ...
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    ...
]
```

Typically, the data augmentations in the data pipeline handles only video-level transforms, such as resizing or cropping, but not transforms like video normalization or mixup/cutmix. This is because we can do video normalization and mixup/cutmix on batched video data
to accelerate processing using GPUs. To configure video normalization and mixup/cutmix, please use the [mmaction.models.utils.data_preprocessor](mmaction.models.utils.data_preprocessor).

### Formatting

Formatting involves collecting training data from the data information dict and converting it into a format that is compatible with the model.

In most cases, you can simply employ [`PackActionInputs`](mmaction.datasets.transforms.PackActionInputs), and it will
convert the image in `NumPy Array` format to `PyTorch Tensor`, and pack the ground truth category information and
other meta information as a dict-like object [`ActionDataSample`](mmaction.structures.ActionDataSample).

```python
train_pipeline = [
    ...
    dict(type='PackActionInputs'),
]
```

## Add New Data Transforms

1. To create a new data transform, write a new transform class in a python file named, for example, `my_transforms.py`. The data transform classes must inherit
   the [`mmcv.transforms.BaseTransform`](mmcv.transforms.BaseTransform) class and override the `transform` method which takes a `dict` as input and returns a `dict`. Finally, place `my_transforms.py` in the folder `mmaction/datasets/transforms/`.

   ```python
   from mmcv.transforms import BaseTransform
   from mmaction.datasets import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):
        def __init__(self, msg):
            self.msg = msg

       def transform(self, results):
           # Modify the data information dict `results`.
           print(msg, 'MMAction2.')
           return results
   ```

2. Import the new class in the `mmaction/datasets/transforms/__init__.py`.

   ```python
   ...
   from .my_transform import MyTransform

   __all__ = [
       ..., 'MyTransform'
   ]
   ```

3. Use it in config files.

   ```python
   train_pipeline = [
       ...
       dict(type='MyTransform', msg='Hello!'),
       ...
   ]
   ```
