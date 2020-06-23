# Technical Details

In this section, we will introduce the three main units of training a recognizer or localizer:
data pipeline, model and iteration pipeline.

## Data pipeline

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers.
In the data loader, a data preparation pipeline is defined to pre-process data.
At the end of the data preparation pipeline, a dict of data items corresponding
the arguments of models' forward method is returned, and it will be fed into the model.

> Since the data in action recognition & localization may not be the same size (image size, gt bbox size, etc.), the `DataContainer` type in MMCV is used to help collect and distribute data of different size. See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decoupled.
Usually a dataset
defines how to process the annotations while a data pipeline defines all the steps to prepare a data dict.
A data preparation pipeline consists of a sequence of operations.
Each operation takes a dict as input and also output a dict for the next transformation.

A typical pipeline is shown in the following figure.
With the pipeline going on, each operator can add new keys (marked as green) to the result dict or update the existing keys (marked as orange).
![pipeline figure](imgs/data_pipeline.png)

The operations are categorized into data loading, pre-processing, formatting.

Here is a pipeline example for TSN.
```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', `clip_len`=1, `frame_interval`=1, `num_clips`=3),
    dict(type='FrameSelector', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        `scales`=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), `keep_ratio`=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['`imgs`', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['`imgs`', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        `clip_len`=1,
        `frame_interval`=1,
        `num_clips`=3,
        test_mode=True),
    dict(type='FrameSelector', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['`imgs`', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['`imgs`'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        `clip_len`=1,
        `frame_interval`=1,
        `num_clips`=25,
        test_mode=True),
    dict(type='FrameSelector', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['`imgs`', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['`imgs`'])
]
```

For each operation, we list the related dict fields that are added/updated/removed.

### Data loading

`SampleFrames`
- add: `frame_inds`, `clip_len`, `frame_interval`, `num_clips`, *`total_frames`

`DenseSampleFrames`
- add: `frame_inds`, `clip_len`, `frame_interval`, `num_clips`, *`total_frames`

`PyAVDecode`
- add: `imgs`, `original_shape`
- update: *`frame_inds`

`DecordDecode`
- add: `imgs`, `original_shape`
- update: *`frame_inds`

`OpenCVDecode`
- add: `imgs`, `original_shape`
- update: *`frame_inds`

`FrameSelector`
- add: `imgs`, `original_shape`
- update: *`frame_inds`

`LoadLocalizationFeature`
- add: `raw_feature`

`GenerateLocalizationLabels`
- add: `gt_bbox`

`LoadProposals`
- add: `bsp_feature`, tmin, tmax, tmin_score, tmax_score, reference_temporal_iou

### Pre-processing

`RandomCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

`RandomResizedCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

`MultiScaleCrop`
- add: `crop_bbox`, `img_shape`, `scales`
- update: `imgs`

`Resize`
- add: `img_shape`, `keep_ratio`, `scale_factor`
- update: `imgs`

`Flip`
- add: flip, `flip_direction`
- update: `imgs`

`Normalize`
- add: img_norm_cfg
- update: `imgs`

`CenterCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

`ThreeCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

`TenCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

`MultiGroupCrop`
- add: `crop_bbox`, `img_shape`
- update: `imgs`

### Formatting

`ToTensor`
- update: specified by `keys`.

`ToDataContainer`
- update: specified by `fields`.

`ImageToTensor`
- update: specified by `keys`.

`Transpose`
- update: specified by `keys`.

`Collect`
- add: `img_meta` (the keys of `img_meta` is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

`FormatShape`
- add: `input_shape`
- update: `imgs`

## Model

In MMAction, model components are basically categorized as 4 types.

- recognizer: the whole recognizer model pipeline, usually contains a backbone and cls_head.
- backbone: usually an FCN network to extract feature maps, e.g., ResNet, BNInception.
- cls_head: the component for classification task, usually contains an FC layer with some pooling layers.
- localizer: the model for localization task, currently available: BSN, BMN.

### Build a model with basic components

Following some basic pipelines (e.g., `Recognizer2D`), the model structure
can be customized through config files with no pains.

If we want to implement some new components, e.g., the temporal shift backbone structure as
in [TSM: Temporal Shift Module for Efficient Video Understanding](https://arxiv.org/abs/1811.08383), there are several things to do.

1. create a new file in `mmaction/models/backbones/resnet_tsm.py`.

  ```python
  from ..registry import BACKBONES
  from .resnet import ResNet

  @BACKBONES.register_module()
  class ResNetTSM(ResNet):

      def __init__(self,
                   depth,
                   num_segments=8,
                   is_shift=True,
                   shift_div=8,
                   shift_place='blockres',
                   temporal_pool=False,
                   **kwargs):
          pass

      def forward(self, x):
          # implementation is ignored
          pass
  ```

2. Import the module in `mmaction/models/backbones/__init__.py`
  ```python
  from .resnet_tsm import ResNetTSM
  ```

3. modify the config file from

  ```python
  backbone=dict(
      type='ResNet',
      pretrained='torchvision://resnet50',
      depth=50,
      norm_eval=False)
  ```

  to

  ```python
  backbone=dict(
        type='ResNetTSM',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False,
        shift_div=8)
  ```

### Write a new model

To write a new recognition pipeline, you need to inherit from `BaseRecognizer`,
which defines the following abstract methods.

- `forward_train()`: forward method of the training mode
- `forward_test()`: forward method of the testing mode

[Recognizer2D](../mmaction/models/recognizers/recognizer2d.py) and [Recognizer3D](../mmaction/models/recognizers/recognizer3d.py)
are good examples which show how to do that.

## Iteration pipeline

We adopt distributed training for both single machine and multiple machines.
Supposing that the server has 8 GPUs, 8 processes will be started and each process runs on a single GPU.

Each process keeps an isolated model, data loader, and optimizer.
Model parameters are only synchronized once at the begining.
After a forward and backward pass, gradients will be allreduced among all GPUs,
and the optimizer will update model parameters.
Since the gradients are allreduced, the model parameter stays the same for all processes after the iteration.
