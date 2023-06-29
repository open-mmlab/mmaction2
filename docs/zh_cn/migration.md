# 从 MMAction2 0.x 迁移

MMAction2 1.x 引入了一些重构和修改，包括一些向后不兼容的更改。我们提供这个教程，帮助您从 MMAction2 0.x 迁移您的项目。

## 新的依赖项

MMAction2 1.x 依赖于以下库。建议您准备一个新的运行环境，并根据[安装教程](./get_started/installation.md)进行安装。

1. [MMEngine](https://github.com/open-mmlab/mmengine)：MMEngine 是引入于 OpenMMLab 2.0 架构中的用于训练深度学习模型的基础库。
2. [MMCV](https://github.com/open-mmlab/mmcv)：MMCV 是用于计算机视觉的基础库。MMAction2 1.x 需要 `mmcv>=2.0.0`，它比 `mmcv-full==2.0.0` 更紧凑和高效。

## 配置文件

在 MMAction2 1.x 中，我们重构了配置文件的结构。旧风格的配置文件将不兼容。

在本节中，我们将介绍配置文件的所有更改。我们假设您已经熟悉[配置文件](./user_guides/config.md)。

### 模型设置

`model.backbone` 和 `model.neck` 没有更改。对于 `model.cls_head`，我们将 `average_clips` 移到其中，原本设置在 `model.test_cfg` 中。

### 数据设置

#### **`data`** 中的更改

- 原始的 `data` 字段被拆分为 `train_dataloader`、`val_dataloader` 和 `test_dataloader`。这样可以对它们进行细粒度的配置。例如，您可以在训练和测试过程中指定不同的采样器和批大小。
- `videos_per_gpu` 改名为 `batch_size`。
- `workers_per_gpu` 改名为 `num_workers`。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
<td>

```python
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=True)  # 必要
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(...),
    sampler=dict(type='DefaultSampler', shuffle=False)  # 必要
)

test_dataloader = val_dataloader
```

</td>
</tr>
</table>

#### **`pipeline`** 中的更改

- 原来的格式化变换 **`ToTensor`**、**`Collect`** 被合并为 `PackActionInputs`。
- 我们不建议在数据集流水线中进行 **`Normalize`**。请从流水线中移除它，并在 `model.data_preprocessor` 字段中设置。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
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

#### **`evaluation`** 中的更改

- **`evaluation`** 字段被拆分为 `val_evaluator` 和 `test_evaluator`。不再支持 `interval` 和 `save_best` 参数。
- `interval` 移到 `train_cfg.val_interval`，`save_best` 移到 `default_hooks.checkpoint.save_best`。
- 'mean_average_precision'、'mean_class_accuracy'、'mmit_mean_average_precision'、'top_k_accuracy' 被合并为 `AccMetric`，您可以使用 `metric_list` 指定要计算的指标。
- `AVAMetric` 用于评估 AVA 数据集。
- `ANetMetric` 用于评估 ActivityNet 数据集。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy'])
```

</td>
<tr>
<td>新版本</td>
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

### 学习率策略设置

#### **`optimizer`** 和 **`optimizer_config`** 中的更改

- 现在我们使用 `optim_wrapper` 字段来配置优化过程。`optimizer` 成为 `optim_wrapper` 的子字段。
- `paramwise_cfg` 也是 `optim_wrapper` 的子字段，与 `optimizer` 平行。
- 现在已删除 `optimizer_config`，其中的所有配置都移动到 `optim_wrapper`。
- `grad_clip` 改名为 `clip_grad`。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
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

#### **`lr_config`** 中的更改

- 删除了 `lr_config` 字段，我们使用新的 `param_scheduler` 来替代它。
- 删除了与 warmup 相关的参数，因为我们使用策略组合来实现这个功能。

新的组合机制非常灵活，您可以使用它来设计多种学习率/动量曲线。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
<td>

```python
param_scheduler = [
    # 学习率预热
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=5,
        # 在每个迭代后更新学习率。
        convert_to_iter_based=True),
    # 主要的学习率策略
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5),
]
```

</td>
</tr>
</table>

#### **`runner`** 中的更改

原始 `runner` 字段中的大多数配置已移至 `train_cfg`、`val_cfg` 和 `test_cfg`，用于配置训练、验证和测试的循环。

<table class="docutils">
<tr>
<td>旧版本</td>
<td>

```python
runner = dict(type='EpochBasedRunner', max_epochs=100)
```

</td>
<tr>
<td>新版本</td>
<td>

```python
# `val_interval` 是原 `evaluation.interval`。
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')   # 使用默认验证循环。
test_cfg = dict(type='TestLoop')  # 使用默认测试循环。
```

</td>
</tr>
</table>

事实上，在 OpenMMLab 2.0 中，我们引入了 `Loop` 来控制训练、验证和测试的行为。`Runner` 的功能也发生了变化。您可以在[MMEngine 教程](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html)中找到更多详细信息。

### 运行时设置

#### **`checkpoint_config`** 和 **`log_config`** 中的更改

`checkpoint_config` 移到 `default_hooks.checkpoint`，`log_config` 移到 `default_hooks.logger`。我们将许多钩子的设置从脚本代码中移动到运行时配置的 `default_hooks` 字段中。

```python
default_hooks = dict(
    # 更新运行时信息，如当前迭代和学习率。
    runtime_info=dict(type='RuntimeInfoHook'),

    # 记录每个迭代的时间。
    timer=dict(type='IterTimerHook'),

    # 每 100 次迭代打印日志。
    logger=dict(type='LoggerHook', interval=100),

    # 启用参数策略器。
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 每个 epoch 保存一次权重，并自动保存最佳权重。
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),

    # 在分布式环境中设置采样器种子。
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # 在每个 epoch 结束时同步模型缓冲区。
    sync_buffers=dict(type='SyncBuffersHook')
)
```

此外，我们将原来的 logger 拆分为 logger 和 visualizer。logger 用于记录信息，visualizer 用于在不同的后端（如终端、TensorBoard 和 Wandb）中显示 logger。

<table class="docutils">
<tr>
<td>旧版本</td>
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
<td>新版本</td>
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

#### **`load_from`** 和 **`resume_from`** 中的更改

- 删除了 `resume_from`。现在我们使用 `resume` 和 `load_from` 来替代它。
  - 如果 `resume=True` 并且 `load_from` 不为 None，则从 `load_from` 中的权重恢复训练。
  - 如果 `resume=True` 并且 `load_from` 为 None，则尝试从工作目录中的最新权重恢复。
  - 如果 `resume=False` 并且 `load_from` 不为 None，则只加载权重文件，不恢复训练。
  - 如果 `resume=False` 并且 `load_from` 为 None，则既不加载也不恢复。

#### **`dist_params`** 中的更改

`dist_params` 字段现在是 `env_cfg` 的子字段。`env_cfg` 中还有一些新的配置。

```python
env_cfg = dict(
    # 是否启用 cudnn benchmark
    cudnn_benchmark=False,

    # 设置多进程参数
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # 设置分布式参数
    dist_cfg=dict(backend='nccl'),
)
```

#### **`workflow`** 中的更改

删除了与 `workflow` 相关的功能。

#### 新字段 **`visualizer`**

visualizer 是 OpenMMLab 2.0 架构中的新设计。我们在 runner 中使用一个 visualizer 实例来处理结果和日志的可视化，并保存到不同的后端，如终端、TensorBoard 和 Wandb。

```python
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        # 取消下面一行的注释，将日志和可视化结果保存到 TensorBoard。
        # dict(type='TensorboardVisBackend')
    ]
)
```

#### 新字段 **`default_scope`**

所有注册表在不同包中的定义已移动到 `mmaction.registry` 包中。

## Packages

### `mmaction.apis`

文档可以在[这里](mmaction.apis)找到。

|          函数          |                     更改                     |
| :--------------------: | :------------------------------------------: |
|   `init_recognizer`    |                   无需更改                   |
| `inference_recognizer` |                   无需更改                   |
|     `train_model`      |      删除，使用 `runner.train` 进行训练      |
|    `multi_gpu_test`    |      删除，使用 `runner.test` 进行测试       |
|   `single_gpu_test`    |      删除，使用 `runner.test` 进行测试       |
|   `set_random_seed`    | 删除，使用 `mmengine.runner.set_random_seed` |
|   `init_random_seed`   | 删除，使用 `mmengine.dist.sync_random_seed`  |

### `mmaction.core`

`mmaction.core` 包已被重命名为 [`mmaction.engine`](mmaction.engine)。

|     子包     |                           更改                            |
| :----------: | :-------------------------------------------------------: |
| `evaluation` |         删除，使用 `mmaction.evaluation` 中的指标         |
|   `hooks`    |              移动到 `mmaction.engine.hooks`               |
| `optimizer`  |            移动到 `mmaction.engine.optimizers`            |
|   `utils`    | 删除，分布式环境相关的函数可以在 `mmengine.dist` 包中找到 |

### `mmaction.datasets`

文档可以在[这里](mmaction.datasets)找到。

#### [`BaseActionDataset`](mmaction.datasets.BaseActionDataset) 中的更改：

|          方法          |                    更改                     |
| :--------------------: | :-----------------------------------------: |
| `prepare_train_frames` |           由 `get_data_info` 替换           |
| `preprare_test_frames` |           由 `get_data_info` 替换           |
|       `evaluate`       |  删除，使用 `mmengine.evaluator.Evaluator`  |
|     `dump_results`     | 删除，使用 `mmengine.evaluator.DumpResults` |
|   `load_annotations`   |           替换为 `load_data_list`           |

现在，您可以编写一个继承自 `BaseActionDataset` 的新 Dataset 类，并仅重写 `load_data_list`。要加载更多的数据信息，您可以像 `RawframeDataset` 和 `AVADataset` 那样重写 `get_data_info`。
`mmaction.datasets.pipelines` 被重命名为 `mmaction.datasets.transforms`，`mmaction.datasets.pipelines.augmentations` 被重命名为 `mmaction.datasets.pipelines.processing`。

### `mmaction.models`

文档可以在[这里](mmaction.models)找到。所有 **backbones**、**necks** 和 **losses** 的接口没有更改。

[`BaseRecognizer`](mmaction.models.BaseRecognizer) 中的更改：

|      方法       |                                                              更改                                                              |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| `extract_feat`  | 增强的方法，现在支持三个阶段（`backbone`、`neck`、`head`）的输出特征，并且可以处理不同的模式，如 `train_mode` 和 `test_mode`。 |
|    `forward`    |         现在只接受三个参数：`inputs`、`data_samples` 和 `mode`。详细信息请参阅[文档](mmaction.models.BaseRecognizer)。         |
| `forward_train` |                                                       已替换为 `loss`。                                                        |
| `forward_test`  |                                                      已替换为 `predict`。                                                      |
|  `train_step`   |                `optimizer` 参数被替换为 `optim_wrapper`，它接受 [`OptimWrapper`](mmengine.optim.OptimWrapper)。                |
|   `val_step`    |                                    原 `val_step` 与 `train_step` 相同，现在调用 `predict`。                                    |
|   `test_step`   |                                                  新方法，与 `val_step` 相同。                                                  |

[BaseHead](mmaction.models.BaseHead) 中的更改：

|   方法    |                                                                              更改                                                                              |
| :-------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| `forward` |                                                                            无需更改                                                                            |
|  `loss`   | 接受 `feats` 和 `data_samples`，而不是 `cls_score` 和 `labels` 来计算损失。`data_samples` 是 [ActionDataSample](mmaction.structures.ActionDataSample) 的列表。 |
| `predict` |                                                        接受 `feats` 和 `data_samples` 来预测分类分数。                                                         |

### `mmaction.utils`

|          函数           |                            更改                            |
| :---------------------: | :--------------------------------------------------------: |
|      `collect_env`      |                          无需更改                          |
|    `get_root_logger`    |    删除，使用 `mmengine.MMLogger.get_current_instance`     |
| `setup_multi_processes` | 删除，使用 `mmengine.utils.dl_utils.setup_multi_processes` |

### 其他更改

- 我们将所有注册器的定义从各个包移动到了 `mmaction.registry` 。
