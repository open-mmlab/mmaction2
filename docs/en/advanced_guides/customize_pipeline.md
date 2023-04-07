# Customize Data Pipeline

In this tutorial, we will introduce some methods about how to build the data pipeline (i.e., data transformations)for your tasks.

- [Customize Data Pipeline](#customize-data-pipeline)
  - [Design of Data pipelines](#design-of-data-pipelines)
  - [Modify the training/test pipeline](#modify-the-trainingtest-pipeline)
    - [Loading](#loading)
    - [Sampling frames and other processing](#sampling-frames-and-other-processing)
    - [Formatting](#formatting)
  - [Add new data transforms](#add-new-data-transforms)

## Design of Data pipelines

The data pipeline means how to process the sample dict when indexing a sample from the dataset. And it
consists of a sequence of data transforms. Each data transform takes a dict as input, processes it, and outputs a dict for the next data transform.

Here is a data pipeline example for SlowFast training on Kinetics for `VideoDataset`. It first use [`decord`](https://github.com/dmlc/decord) to read the raw videos and randomly sample one video clip (the clip has 32 frames, and the interval between frames is 2). Next it applies the random resized crop and random horizontal flip to all frames. Finally the data shape is formatted as `NCTHW`.

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

All available data transforms in MMAction2 can be found in the [data transforms docs](mmaction.datasets.transforms).

## Modify the training/test pipeline

The data pipeline in MMAction2 is pretty flexible. You can control almost every step of the data
preprocessing from the config file, but on the other hand, you may be confused facing so many options.

Here is a common practice and guidance for action recognition tasks.

### Loading

At the beginning of a data pipeline, we usually need to load videos. But if you already extract the frames, you should use `RawFrameDecode` and change the dataset type to `RawframeDataset`:

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

If you want to load data from files with special formats or special locations, you can [implement a new loading
transform](#add-new-data-transforms) and add it at the beginning of the data pipeline.

### Sampling frames and other processing

During training and testing, we may have different strategies to sample frames from the video.

For example, during testing of SlowFast, we sample multiple clips uniformly:

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

In the above example, 10 clips of 32-frame video clips will be sampled for each video. We use `test_mode=True` to uniformly sample these clips (as opposed to randomly sample during training).

Another example is that TSN/TSM models sample multiple segments from the video:

```python
train_pipeline = [
    ...
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    ...
]
```

```{note}
Usually, the data augmentation part in the data pipeline handles only video-wise transforms, but not transforms
like video normalization or mixup/cutmix. It's because we can do image normalization and mixup/cutmix on batch data
to accelerate with GPUs. To configure video normalization and mixup/cutmix, please use the [data preprocessor]
(mmaction.models.utils.data_preprocessor).
```

### Formatting

The formatting is to collect training data from the data information dict and convert these data to
model-friendly format.

In most cases, you can simply use [`PackActionInputs`](mmaction.datasets.transforms.PackActionInputs), and it will
convert the image in NumPy array format to PyTorch tensor, and pack the ground truth categories information and
other meta information as a dict-like object [`ActionDataSample`](mmaction.structures.ActionDataSample).

```python
train_pipeline = [
    ...
    dict(type='PackActionInputs'),
]
```

## Add new data transforms

1. Write a new data transform in any file, e.g., `my_transform.py`, and place it in
   the folder `mmaction/datasets/transforms/`. The data transform class needs to inherit
   the [`mmcv.transforms.BaseTransform`](mmcv.transforms.BaseTransform) class and override
   the `transform` method which takes a dict as input and returns a dict.

   ```python
   from mmcv.transforms import BaseTransform
   from mmaction.datasets import TRANSFORMS

   @TRANSFORMS.register_module()
   class MyTransform(BaseTransform):

       def transform(self, results):
           # Modify the data information dict `results`.
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
       dict(type='MyTransform'),
       ...
   ]
   ```
