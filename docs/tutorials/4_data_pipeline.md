# Tutorial 4: Customize Data Pipelines

In this tutorial, we will introduce some methods about the design of data pipelines, and how to customize and extend your own data pipelines for the project.

<!-- TOC -->

- [Tutorial 4: Customize Data Pipelines](#tutorial-4-customize-data-pipelines)
  - [Design of Data Pipelines](#design-of-data-pipelines)
    - [Data loading](#data-loading)
    - [Pre-processing](#pre-processing)
    - [Formatting](#formatting)
  - [Extend and Use Custom Pipelines](#extend-and-use-custom-pipelines)

<!-- TOC -->

## Design of Data Pipelines

Following typical conventions, we use `Dataset` and `DataLoader` for data loading
with multiple workers. `Dataset` returns a dict of data items corresponding
the arguments of models' forward method.
Since the data in action recognition & localization may not be the same size (image size, gt bbox size, etc.),
The `DataContainer` in MMCV is used to help collect and distribute data of different sizes.
See [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py) for more details.

The data preparation pipeline and the dataset is decomposed. Usually a dataset
defines how to process the annotations and a data pipeline defines all the steps to prepare a data dict.
A pipeline consists of a sequence of operations. Each operation takes a dict as input and also output a dict for the next operation.

We present a typical pipeline in the following figure. The blue blocks are pipeline operations.
With the pipeline going on, each operator can add new keys (marked as green) to the result dict or update the existing keys (marked as orange).
![pipeline figure](https://github.com/open-mmlab/mmaction2/raw/master/resources/data_pipeline.png)

The operations are categorized into data loading, pre-processing and formatting.

Here is a pipeline example for TSN.

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='RawFrameDecode', io_backend='disk'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='TenCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
```

We have supported some lazy operators and encourage users to apply them.
Lazy ops record how the data should be processed, but it will postpone the processing on the raw data until the raw data forward `Fuse` stage.
Specifically, lazy ops avoid frequent reading and modification operation on the raw data, but process the raw data once in the final Fuse stage, thus accelerating data preprocessing.

Here is a pipeline example applying lazy ops.

```python
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode', decoding_backend='turbojpeg'),
    # The following three lazy ops only process the bbox of frames without
    # modifying the raw data.
    dict(type='Resize', scale=(-1, 256), lazy=True),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0,
        lazy=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=False, lazy=True),
    # Lazy operator `Flip` only record whether a frame should be fliped and the
    # flip direction.
    dict(type='Flip', flip_ratio=0.5, lazy=True),
    # Processing the raw data once in Fuse stage.
    dict(type='Fuse'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
```

For each operation, we list the related dict fields that are added/updated/removed below, where `*` means the key may not be affected.

### Data loading

`SampleFrames`

- add: frame_inds, clip_len, frame_interval, num_clips, *total_frames

`DenseSampleFrames`

- add: frame_inds, clip_len, frame_interval, num_clips, *total_frames

`PyAVDecode`

- add: imgs, original_shape
- update: *frame_inds

`DecordDecode`

- add: imgs, original_shape
- update: *frame_inds

`OpenCVDecode`

- add: imgs, original_shape
- update: *frame_inds

`RawFrameDecode`

- add: imgs, original_shape
- update: *frame_inds

### Pre-processing

`RandomCrop`

- add: crop_bbox, img_shape
- update: imgs

`RandomResizedCrop`

- add: crop_bbox, img_shape
- update: imgs

`MultiScaleCrop`

- add: crop_bbox, img_shape, scales
- update: imgs

`Resize`

- add: img_shape, keep_ratio, scale_factor
- update: imgs

`Flip`

- add: flip, flip_direction
- update: imgs, label

`Normalize`

- add: img_norm_cfg
- update: imgs

`CenterCrop`

- add: crop_bbox, img_shape
- update: imgs

`ThreeCrop`

- add: crop_bbox, img_shape
- update: imgs

`TenCrop`

- add: crop_bbox, img_shape
- update: imgs

### Formatting

`ToTensor`

- update: specified by `keys`.

`ImageToTensor`

- update: specified by `keys`.

`Transpose`

- update: specified by `keys`.

`Collect`

- add: img_metas (the keys of img_metas is specified by `meta_keys`)
- remove: all other keys except for those specified by `keys`

It is **noteworthy** that the first key, commonly `imgs`, will be used as the main key to calculate the batch size.

`FormatShape`

- add: input_shape
- update: imgs

## Extend and Use Custom Pipelines

1. Write a new pipeline in any file, e.g., `my_pipeline.py`. It takes a dict as input and return a dict.

    ```python
    from mmaction.datasets import PIPELINES

    @PIPELINES.register_module()
    class MyTransform:

        def __call__(self, results):
            results['key'] = value
            return results
    ```

2. Import the new class.

    ```python
    from .my_pipeline import MyTransform
    ```

3. Use it in config files.

    ```python
    img_norm_cfg = dict(
         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    train_pipeline = [
        dict(type='DenseSampleFrames', clip_len=8, frame_interval=8, num_clips=1),
        dict(type='RawFrameDecode', io_backend='disk'),
        dict(type='MyTransform'),       # use a custom pipeline
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ]
    ```
