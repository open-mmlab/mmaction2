# Migration from MMAction2 0.x

MMAction2 1.x introduced major refactorings and modifications including some BC-breaking changes. We provide this tutorial to help you migrate your projects from MMAction2 0.x smoothly.

## New dependencies

MMAction2 1.x depends on the following packages. You are recommended to prepare a new clean environment and install them according to [install tutorial](./get_started/installation.md)

1. [MMEngine](https://github.com/open-mmlab/mmengine): MMEngine is a foundational library for training deep learning model introduced in OpenMMLab 2.0 architecture.
2. [MMCV](https://github.com/open-mmlab/mmcv): MMCV is a foundational library for computer vision. MMAction2 1.x requires `mmcv>=2.0.0` which is more compact and efficient than `mmcv-full==2.0.0`.

## Configuration files

In MMAction2 1.x, we refactored the structure of configuration files. The configuration files with the old style will be incompatible.

In this section, we will introduce all changes of the configuration files. And we assume you are already familiar with the [config files](./user_guides/config.md).

### Model settings

No changes in `model.backbone` and `model.neck`. For `model.cls_head`, we move the `average_clips` inside it, which is originally set in `model.test_cfg`.

### Data settings

#### Changes in **`data`**

- The original `data` field is splited to `train_dataloader`, `val_dataloader` and
  `test_dataloader`. This allows us to configure them in fine-grained. For example,
  you can specify different sampler and batch size during training and test.
- The `videos_per_gpu` is renamed to `batch_size`.
- The `workers_per_gpu` is renamed to `num_workers`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    train=dict(...),
    val=dict(...),
    test=dict(...),
)
```

</td>
<tr>
<td>New</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # necessary
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # necessary
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

#### Changes in **`pipeline`**

- The original formatting transforms **`ToTensor`**, **`Collect`** are combined as `PackActionInputs`.
- We don't recommend to do **`Normalize`** in the dataset pipeline. Please remove it from pipelines and set it in the `model.data_preprocessor` field.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
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
```

</td>
<tr>
<td>New</td>
<td>

```python
model.data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
```

</td>
</tr>
</table>

#### Changes in **`evaluation`**

- The **`evaluation`** field is splited to `val_evaluator` and `test_evaluator`. And it won't support `interval` and `save_best` arguments.
- The `interval` is moved to `train_cfg.val_interval` and the `save_best` is moved to `default_hooks.checkpoint.save_best`.
- The 'mean_average_precision', 'mean_class_accuracy', 'mmit_mean_average_precision', 'top_k_accuracy' are combined as `AccMetric`, and you could use `metric_list` to specify which metric to calculate.
- The `AVAMetric` is used to evaluate AVA Dataset.
- The `ANetMetric` is used to evaluate ActivityNet Dataset.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])
```

</td>
<tr>
<td>New</td>
<td>

```python
val_evaluator = dict(
    type='AccMetric',
    metric_list=('top_k_accuracy', 'mean_class_accuracy'))
test_evaluator = val_evaluator
```

</td>
</tr>
</table>

### Schedule settings

#### Changes in **`optimizer`** and **`optimizer_config`**

- Now we use `optim_wrapper` field to configure the optimization process. And the
  `optimizer` becomes a sub field of `optim_wrapper`.
- `paramwise_cfg` is also a sub field of `optim_wrapper` parallel to `optimizer`.
- `optimizer_config` is removed now, and all configurations of it are moved to `optim_wrapper`.
- `grad_clip` is renamed to `clip_grad`.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
optimizer = dict(
    type='AdamW',
    lr=0.0015,
    weight_decay=0.3,
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ))

optimizer_config = dict(grad_clip=dict(max_norm=1.0))
```

</td>
<tr>
<td>New</td>
<td>

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0015, weight_decay=0.3),
    paramwise_cfg = dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
    ),
    clip_gard=dict(max_norm=1.0),
)
```

</td>
</tr>
</table>

#### Changes in **`lr_config`**

- The `lr_config` field is removed and we use new `param_scheduler` to replace it.
- The `warmup` related arguments are removed, since we use schedulers combination to implement this
  functionality.

The new schedulers combination mechanism is very flexible, and you can use it to design many kinds of learning
rate / momentum curves.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
```

</td>
<tr>
<td>New</td>
<td>

```python
param_scheduler = [
    # warmup
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        # Update the learning rate after every iters.
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5),
]
```

</td>
</tr>
</table>

#### Changes in **`runner`**

Most configuration in the original `runner` field is moved to `train_cfg`, `val_cfg` and `test_cfg`, which
configure the loop in training, validation and test.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=100)
```

</td>
<tr>
<td>New</td>
<td>

```python
# The `val_interval` is the original `evaluation.interval`.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')   # Use the default validation loop.
test_cfg = dict(type='TestLoop')  # Use the default test loop.
```

</td>
</tr>
</table>

In fact, in OpenMMLab 2.0, we introduced `Loop` to control the behaviors in training, validation and test. And
the functionalities of `Runner` are also changed. You can find more details in the [MMEngine tutorials](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html).

### Runtime settings

#### Changes in **`checkpoint_config`** and **`log_config`**

The `checkpoint_config` are moved to `default_hooks.checkpoint` and the `log_config` are moved to `default_hooks.logger`.
And we move many hooks settings from the script code to the `default_hooks` field in the runtime configuration.

```python
default_hooks = dict(
    # update runtime information, e.g. current iter and lr.
    runtime_info=dict(type='RuntimeInfoHook'),

    # record the time of every iterations.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch, and automatically save the best checkpoint.
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),

    # set sampler seed in distributed environment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # synchronize model buffers at the end of each epoch.
    sync_buffers=dict(type='SyncBuffersHook')
)
```

In addition, we splited the original logger to logger and visualizer. The logger is used to record
information and the visualizer is used to show the logger in different backends, like terminal, TensorBoard
and Wandb.

<table class="docutils">
<tr>
<td>Original</td>
<td>

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
```

</td>
<tr>
<td>New</td>
<td>

```python
default_hooks = dict(
    ...
    logger=dict(type='LoggerHook', interval=100),
)

visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
)
```

</td>
</tr>
</table>

#### Changes in **`load_from`** and **`resume_from`**

- The `resume_from` is removed. And we use `resume` and `load_from` to replace it.
  - If `resume=True` and `load_from` is not None, resume training from the checkpoint in `load_from`.
  - If `resume=True` and `load_from` is None, try to resume from the latest checkpoint in the work directory.
  - If `resume=False` and `load_from` is not None, only load the checkpoint, not resume training.
  - If `resume=False` and `load_from` is None, do not load nor resume.

#### Changes in **`dist_params`**

The `dist_params` field is a sub field of `env_cfg` now. And there are some new configurations in the `env_cfg`.

```python
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)
```

#### Changes in **`workflow`**

`Workflow` related functionalities are removed.

#### New field **`visualizer`**

The visualizer is a new design in OpenMMLab 2.0 architecture. We use a visualizer instance in the runner to handle results & log visualization and save to different backends.

```python
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # Uncomment the below line to save the log and visualization results to TensorBoard.
        # dict(type='TensorboardVisBackend')
    ]
)
```

#### New field **`default_scope`**

The start point to search module for all registries. The `default_scope` in MMAction2 is `mmaction`. See [the registry tutorial](https://mmengine.readthedocs.io/en/latest/tutorials/registry.html) for more details.

## Packages

### `mmaction.apis`

The documentation can be found [here](mmaction.apis).

|        Function        |                     Changes                     |
| :--------------------: | :---------------------------------------------: |
|   `init_recognizer`    |                   No changes                    |
| `inference_recognizer` |                   No changes                    |
|     `train_model`      |      Removed, use `runner.train` to train.      |
|    `multi_gpu_test`    |       Removed, use `runner.test` to test.       |
|   `single_gpu_test`    |       Removed, use `runner.test` to test.       |
|   `set_random_seed`    | Removed, use `mmengine.runner.set_random_seed`. |
|   `init_random_seed`   | Removed, use `mmengine.dist.sync_random_seed`.  |

### `mmaction.core`

The `mmaction.core` package is renamed to [`mmaction.engine`](mmaction.engine).

| Sub package  |                                               Changes                                               |
| :----------: | :-------------------------------------------------------------------------------------------------: |
| `evaluation` |                         Removed, use the metrics in `mmaction.evaluation`.                          |
|   `hooks`    |                                  Moved to `mmaction.engine.hooks`                                   |
| `optimizer`  |                                Moved to `mmaction.engine.optimizers`                                |
|   `utils`    | Removed, the distributed environment related functions can be found in the `mmengine.dist` package. |

### `mmaction.datasets`

The documentation can be found [here](mmaction.datasets)

#### Changes in [`BaseActionDataset`](mmaction.datasets.BaseActionDataset):

|         Method         |                    Changes                    |
| :--------------------: | :-------------------------------------------: |
| `prepare_train_frames` |          Replaced by `get_data_info`          |
| `preprare_test_frames` |          Replaced by `get_data_info`          |
|       `evaluate`       |  Removed, use `mmengine.evaluator.Evaluator`  |
|     `dump_results`     | Removed, use `mmengine.evaluator.DumpResults` |
|   `load_annotations`   |         Replaced by `load_data_list`          |

Now, you can write a new Dataset class inherited from `BaseActionDataset` and overwrite `load_data_list` only. To load more data information, you could overwrite `get_data_info` like `RawframeDataset` and `AVADataset`.
The `mmaction.datasets.pipelines` is renamed to `mmaction.datasets.transforms` and the `mmaction.datasets.pipelines.augmentations` is renamed to `mmaction.datasets.pipelines.processing`.

### `mmaction.models`

The documentation can be found [here](mmaction.models). The interface of all **backbones**, **necks** and **losses** didn't change.

#### Changes in [`BaseRecognizer`](mmaction.models.BaseRecognizer):

|     Method      |                                                                                Changes                                                                                 |
| :-------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| `extract_feat`  | Enhanced method, which now supports output features of three stages (`backbone`, `neck`, `head`) and can handle different modes, such as `train_mode` and `test_mode`. |
|    `forward`    |            Now only accepts three arguments: `inputs`, `data_samples` and `mode`. See [the documentation](mmaction.models.BaseRecognizer) for more details.            |
| `forward_train` |                                                                          Replaced by `loss`.                                                                           |
| `forward_test`  |                                                                         Replaced by `predict`.                                                                         |
|  `train_step`   |                         The `optimizer` argument is replaced by `optim_wrapper` and it accepts [`OptimWrapper`](mmengine.optim.OptimWrapper).                          |
|   `val_step`    |                                              The original `val_step` is the same as `train_step`, now it calls `predict`.                                              |
|   `test_step`   |                                                              New method, and it's the same as `val_step`.                                                              |

#### Changes in [BaseHead](mmaction.models.BaseHead):

|  Method   |                                                                                        Changes                                                                                         |
| :-------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| `forward` |                                                                                       No changes                                                                                       |
|  `loss`   | It accepts `feats` and `data_samples` instead of `cls_score` and `labels` to calculate loss. The `data_samples` is a list of [ActionDataSample](mmaction.structures.ActionDataSample). |
| `predict` |                                                  New method. It accepts `feats` and `data_samples` to predict classification scores.                                                   |

### `mmaction.utils`

|        Function         |                            Changes                            |
| :---------------------: | :-----------------------------------------------------------: |
|      `collect_env`      |                          No changes                           |
|    `get_root_logger`    |     Removed, use `mmengine.MMLogger.get_current_instance`     |
| `setup_multi_processes` | Removed, use `mmengine.utils.dl_utils.setup_multi_processes`. |

### Other changes

- We moved the definition of all registries in different packages to the `mmaction.registry` package.
