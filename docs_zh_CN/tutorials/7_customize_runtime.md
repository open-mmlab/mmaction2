# 教程 7：如何自定义模型运行参数

在本教程中，我们将介绍如何在运行自定义模型时，进行自定义参数优化方法，学习率调整策略，工作流和钩子的方法。

<!-- TOC -->

- [定制优化方法](#定制优化方法)
  - [使用 PyTorch 内置的优化器](#使用-PyTorch-内置的优化器)
  - [定制用户自定义的优化器](#定制用户自定义的优化器)
    - [1. 定义一个新的优化器](#1-定义一个新的优化器)
    - [2. 注册优化器](#2-注册优化器)
    - [3. 在配置文件中指定优化器](#3-在配置文件中指定优化器)
  - [定制优化器构造器](#定制优化器构造器)
  - [额外设定](#额外设定)
- [定制学习率调整策略](#定制学习率调整策略)
- [定制工作流](#定制工作流)
- [定制钩子](#定制钩子)
  - [定制用户自定义钩子](#定制用户自定义钩子)
    - [1. 创建一个新钩子](#1-创建一个新钩子)
    - [2. 注册新钩子](#2-注册新钩子)
    - [3. 修改配置](#3-修改配置)
  - [使用 MMCV 内置钩子](#使用-MMCV-内置钩子)
  - [修改默认运行的钩子](#修改默认运行的钩子)
    - [模型权重文件配置](#模型权重文件配置)
    - [日志配置](#日志配置)
    - [验证配置](#验证配置)

<!-- TOC -->

## 定制优化方法

### 使用 PyTorch 内置的优化器

MMAction2 支持 PyTorch 实现的所有优化器，仅需在配置文件中，指定 “optimizer” 字段
例如，如果要使用 “Adam”，则修改如下。

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

要修改模型的学习率，用户只需要在优化程序的配置中修改 “lr” 即可。
用户可根据 [PyTorch API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 进行参数设置

例如，如果想使用 `Adam` 并设置参数为 `torch.optim.Adam(parms, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)`，
则需要进行如下修改

```python
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

### 定制用户自定义的优化器

#### 1. 定义一个新的优化器

一个自定义的优化器可根据如下规则进行定制

假设用户想添加一个名为 `MyOptimzer` 的优化器，其拥有参数 `a`, `b` 和 `c`，
可以创建一个名为 `mmaction/core/optimizer` 的文件夹，并在目录下的文件进行构建，如 `mmaction/core/optimizer/my_optimizer.py`：

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):

```

#### 2. 注册优化器

要找到上面定义的上述模块，首先应将此模块导入到主命名空间中。有两种方法可以实现它。

- 修改 `mmaction/core/optimizer/__init__.py` 来进行调用

    新定义的模块应导入到 `mmaction/core/optimizer/__init__.py` 中，以便注册器能找到新模块并将其添加：

```python
from .my_optimizer import MyOptimizer
```

- 在配置中使用 `custom_imports` 手动导入

```python
custom_imports = dict(imports=['mmaction.core.optimizer.my_optimizer'], allow_failed_imports=False)
```

`mmaction.core.optimizer.my_optimizer` 模块将会在程序开始阶段被导入，`MyOptimizer` 类会随之自动被注册。
注意，只有包含 `MyOptmizer` 类的包会被导入。`mmaction.core.optimizer.my_optimizer.MyOptimizer` **不会** 被直接导入。

#### 3. 在配置文件中指定优化器

之后，用户便可在配置文件的 `optimizer` 域中使用 `MyOptimizer`。
在配置中，优化器由 “optimizer” 字段定义，如下所示：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

要使用自定义的优化器，可以将该字段更改为

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

### 定制优化器构造器

某些模型可能具有一些特定于参数的设置以进行优化，例如 BatchNorm 层的权重衰减。
用户可以通过自定义优化器构造函数来进行那些细粒度的参数调整。

```python
from mmcv.runner.optimizer import OPTIMIZER_BUILDERS


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        pass

    def __call__(self, model):

        return my_optimizer
```

默认的优化器构造器被创建于[此](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/optimizer/default_constructor.py#L11)，
可被视为新优化器构造器的模板。

### 额外设定

优化器没有实现的优化技巧（trick）可通过优化器构造函数（例如，设置按参数的学习率）或钩子来实现。
下面列出了一些可以稳定训练或加快训练速度的常用设置。用户亦可通过为 MMAction2 创建 PR，发布更多设置。

- __使用梯度裁剪来稳定训练__
    一些模型需要使用梯度裁剪来剪辑渐变以稳定训练过程。 一个例子如下：

    ```python
    optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
    ```

- __使用动量调整来加速模型收敛__
    MMAction2 支持动量调整器根据学习率修改模型的动量，从而使模型收敛更快。
    动量调整程序通常与学习率调整器一起使用，例如，以下配置用于3D检测以加速收敛。
    更多细节可参考 [CyclicLrUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327)
    和 [CyclicMomentumUpdater](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130)。

    ```python
    lr_config = dict(
        policy='cyclic',
        target_ratio=(10, 1e-4),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    momentum_config = dict(
        policy='cyclic',
        target_ratio=(0.85 / 0.95, 1),
        cyclic_times=1,
        step_ratio_up=0.4,
    )
    ```

## 定制学习率调整策略

在配置文件中使用默认值的逐步学习率调整，它调用 MMCV 中的 [`StepLRHook`](https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L153)。
此外，也支持其他学习率调整方法，如 `CosineAnnealing` 和 `Poly`。 详情可见 [这里](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)

- Poly:

    ```python
    lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
    ```

- ConsineAnnealing:

    ```python
    lr_config = dict(
        policy='CosineAnnealing',
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=1.0 / 10,
        min_lr_ratio=1e-5)
    ```

## 定制工作流

默认情况下，MMAction2 推荐用户在训练周期中使用 “EvalHook” 进行模型验证，也可以选择 “val” 工作流模型进行模型验证。

工作流是一个形如 (工作流名, 周期数) 的列表，用于指定运行顺序和周期。其默认设置为：

```python
workflow = [('train', 1)]
```

其代表要进行一轮周期的训练。
有时，用户可能希望检查有关验证集中模型的某些指标（例如，损失，准确性）。
在这种情况下，可以将工作流程设置为

```python
[('train', 1), ('val', 1)]
```

从而将迭代运行1个训练时间和1个验证时间。

**值得注意的是**：

1. 在验证周期时不会更新模型参数。
2. 配置文件内的关键词 `total_epochs` 控制训练时期数，并且不会影响验证工作流程。
3. 工作流 `[('train', 1), ('val', 1)]` 和 `[('train', 1)]` 不会改变 `EvalHook` 的行为。
   因为 `EvalHook` 由 `after_train_epoch` 调用，而验证工作流只会影响 `after_val_epoch` 调用的钩子。
   因此，`[('train', 1), ('val', 1)]` 和 ``[('train', 1)]`` 的区别在于，runner 在完成每一轮训练后，会计算验证集上的损失。

## 定制钩子

### 定制用户自定义钩子

#### 1. 创建一个新钩子

这里举一个在 MMAction2 中创建一个新钩子，并在训练中使用它的示例：

```python
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MyHook(Hook):

    def __init__(self, a, b):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass
```

根据钩子的功能，用户需要指定钩子在训练的每个阶段将要执行的操作，比如 `before_run`，`after_run`，`before_epoch`，`after_epoch`，`before_iter` 和 `after_iter`。

#### 2. 注册新钩子

之后，需要导入 `MyHook`。假设该文件在 `mmaction/core/utils/my_hook.py`，有两种办法导入它：

- 修改 `mmaction/core/utils/__init__.py` 进行导入

    新定义的模块应导入到 `mmaction/core/utils/__init__py` 中，以便注册表能找到并添加新模块：

```python
from .my_hook import MyHook
```

- 使用配置文件中的 `custom_imports` 变量手动导入

```python
custom_imports = dict(imports=['mmaction.core.utils.my_hook'], allow_failed_imports=False)
```

#### 3. 修改配置

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value)
]
```

还可通过 `priority` 参数（可选参数值包括 `'NORMAL'` 或 `'HIGHEST'`）设置钩子优先级，如下所示：

```python
custom_hooks = [
    dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
]
```

默认情况下，在注册过程中，钩子的优先级设置为 “NORMAL”。

### 使用 MMCV 内置钩子

如果该钩子已在 MMCV 中实现，则可以直接修改配置以使用该钩子，如下所示

```python
mmcv_hooks = [
    dict(type='MMCVHook', a=a_value, b=b_value, priority='NORMAL')
]
```

### 修改默认运行的钩子

有一些常见的钩子未通过 `custom_hooks` 注册，但在导入 MMCV 时已默认注册，它们是：

- log_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

在这些钩子中，只有 log_config 具有 “VERY_LOW” 优先级，其他钩子具有 “NORMAL” 优先级。
上述教程已经介绍了如何修改 “optimizer_config”，“momentum_config” 和 “lr_config”。
下面介绍如何使用 log_config，checkpoint_config，以及 evaluation 能做什么。

#### 模型权重文件配置

MMCV 的 runner 使用 `checkpoint_config` 来初始化 [`CheckpointHook`](https://github.com/open-mmlab/mmcv/blob/9ecd6b0d5ff9d2172c49a182eaa669e9f27bb8e7/mmcv/runner/hooks/checkpoint.py#L9)。

```python
checkpoint_config = dict(interval=1)
```

用户可以设置 “max_keep_ckpts” 来仅保存少量模型权重文件，或者通过 “save_optimizer” 决定是否存储优化器的状态字典。
更多细节可参考 [这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook)。

#### 日志配置

`log_config` 包装了多个记录器钩子，并可以设置间隔。
目前，MMCV 支持 `WandbLoggerHook`，`MlflowLoggerHook` 和 `TensorboardLoggerHook`。
更多细节可参考[这里](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook)。

```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
```

#### 验证配置

评估的配置将用于初始化 [`EvalHook`](https://github.com/open-mmlab/mmaction2/blob/master/mmaction/core/evaluation/eval_hooks.py#L12)。
除了键 `interval` 外，其他参数，如 “metrics” 也将传递给 `dataset.evaluate()`。

```python
evaluation = dict(interval=1, metrics='bbox')
```
