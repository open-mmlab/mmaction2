# Config System
We use python files as our config system. You can find all the provided configs under `$MMAction/configs`.

## Config File Naming Convention

We follow the style below to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type, e.g. `tsn`, `i3d`, etc.
- `[model setting]`: specific setting for some models.
- `{backbone}`: backbone type, e.g. `r50` (ResNet-50), etc.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dense`, `320p`, `video`, etc.
- `{data setting}`: frame sample setting in `{clip_len}x{frame_interval}x{num_clips}` format.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU.
- `{schedule}`: training schedule, e.g. `20e` means 20 epochs.
- `{dataset}`: dataset name, e.g. `kinetics400`, `mmit`, etc.
- `{modality}`: frame modality, e.g. `rgb`, `flow`, etc.

## Config File Structure

Please refer to the corresponding pages for config file structure for different tasks.

[Localization](config_localization.md)

[Recognition](config_recognition.md)

## FAQ

### Use intermediate variables in configs

Some intermediate variables are used in the config files, like `train_pipeline`/`val_pipeline`/`test_pipeline`,
`ann_file_train`/`ann_file_val`/`ann_file_test`, `img_norm_cfg` etc.

For Example, we would like to first define `train_pipeline`/`val_pipeline`/`test_pipeline` and pass them into `data`.
Thus, `train_pipeline`/`val_pipeline`/`test_pipeline` are intermediate variable.

we also define `ann_file_train`/`ann_file_val`/`ann_file_test` and `data_root`/`data_root_val` to provide data pipeline some
basic information.

In addition, we use `img_norm_cfg` as intermediate variables to construct data augmentation components.

```python
...
dataset_type = 'RawframeDataset'
data_root = 'data/kinetics400/rawframes_train'
data_root_val = 'data/kinetics400/rawframes_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.8),
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='FrameSelector'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
```
