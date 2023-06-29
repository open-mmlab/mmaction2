# 自定义优化器

在本教程中，我们将介绍一些构建优化器和学习率策略的方法，以用于你的任务。

- [自定义优化器](#自定义优化器)
  - [使用 optim_wrapper 构建优化器](#使用-optim_wrapper-构建优化器)
    - [使用 PyTorch 支持的优化器](#使用-pytorch-支持的优化器)
    - [参数化精细配置](#参数化精细配置)
    - [梯度裁剪](#梯度裁剪)
    - [梯度累积](#梯度累积)
  - [自定义参数策略](#自定义参数策略)
    - [自定义学习率策略](#自定义学习率策略)
    - [自定义动量策略](#自定义动量策略)
  - [添加新的优化器或构造器](#添加新的优化器或构造器)
    - [添加新的优化器](#添加新的优化器)
      - [1. 实现一个新的优化器](#1-实现一个新的优化器)
      - [2. 导入优化器](#2-导入优化器)
      - [3. 在配置文件中指定优化器](#3-在配置文件中指定优化器)
    - [添加新的优化器构造器](#添加新的优化器构造器)

## 使用 optim_wrapper 构建优化器

我们使用 `optim_wrapper` 字段来配置优化策略，其中包括选择优化器、参数逐个配置、梯度裁剪和梯度累积。一个简单的示例可以是：

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0003, weight_decay=0.0001)
)
```

在上面的示例中，我们构建了一个学习率为 0.0003，权重衰减为 0.0001 的 SGD 优化器。

### 使用 PyTorch 支持的优化器

我们支持 PyTorch 实现的所有优化器。要使用不同的优化器，只需更改配置文件中的 `optimizer` 字段。例如，如果想使用 `torch.optim.Adam`，可以在配置文件中进行如下修改。

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer = dict(
        type='Adam',
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False),
)
```

首先，我们需要将 `type` 的值更改为 `torch.optim` 支持的期望优化器名称。然后，将该优化器的必要参数添加到 `optimizer` 字段中。上述配置将构建以下优化器：

```python
torch.optim.Adam(lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 amsgrad=False)
```

### 参数化精细配置

一些模型可能对优化有特定的参数设置，例如对于 BatchNorm 层不使用权重衰减，或者对不同网络层使用不同的学习率。为了对其进行细致配置，我们可以使用 `optim_wrapper` 中的 `paramwise_cfg` 参数。

- **为不同类型的参数设置不同的超参数倍数。**

  例如，我们可以在 `paramwise_cfg` 中设置 `norm_decay_mult=0.`，将归一化层的权重衰减设置为零。

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.8, weight_decay=1e-4),
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

  还支持设置其他类型的参数，包括：

  - `lr_mult`：所有参数的学习率乘数。
  - `decay_mult`：所有参数的权重衰减乘数。
  - `bias_lr_mult`：偏置项的学习率乘数（不包括归一化层的偏置项和可变形卷积层的偏移量）。默认为 1。
  - `bias_decay_mult`：偏置项的权重衰减乘数（不包括归一化层的偏置项和可变形卷积层的偏移量）。默认为 1。
  - `norm_decay_mult`：归一化层权重和偏置项的权重衰减乘数。默认为 1。
  - `dwconv_decay_mult`：深度卷积层的权重衰减乘数。默认为 1。
  - `bypass_duplicate`：是否跳过重复的参数。默认为 `False`。
  - `dcn_offset_lr_mult`：可变形卷积层的学习率乘数。默认为 1。

- **为特定参数设置不同的超参数倍数。**

  MMAction2 可以使用 `paramwise_cfg` 中的 `custom_keys` 来指定不同的参数使用不同的学习率或权重衰减。

  例如，要将 `backbone.layer0` 的所有学习率和权重衰减设置为 0，而保持 `backbone` 的其余部分与优化器相同，并将 `head` 的学习率设置为 0.001，可以使用以下配置：

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
      paramwise_cfg=dict(
          custom_keys={
              'backbone.layer0': dict(lr_mult=0, decay_mult=0),
              'backbone': dict(lr_mult=1),
              'head': dict(lr_mult=0.1)
          }))
  ```

### 梯度裁剪

在训练过程中，损失函数可能接近悬崖区域，导致梯度爆炸。梯度裁剪有助于稳定训练过程。梯度裁剪的更多介绍可以在[这个页面](https://paperswithcode.com/method/gradient-clipping)找到。

目前，我们支持 `optim_wrapper` 中的 `clip_grad` 选项进行梯度裁剪，参考[PyTorch 文档](torch.nn.utils.clip_grad_norm_)。

以下是一个示例：

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    # norm_type: 使用的 p-范数的类型，这里 norm_type 为 2。
    clip_grad=dict(max_norm=35, norm_type=2))
```

### 梯度累积

当计算资源有限时，批量大小只能设置为较小的值，这可能会影响模型的性能。可以使用梯度累积来解决这个问题。我们支持 `optim_wrapper` 中的 `accumulative_counts` 选项进行梯度累积。

以下是一个示例：

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    accumulative_counts=4)
```

表示在训练过程中，每 4 个迭代执行一次反向传播。上述示例等价于：

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001))
```

## 新增优化器或者优化器构造器

在训练中，优化参数（如学习率、动量等）通常不是固定的，而是随着迭代或周期的变化而变化。PyTorch 支持几种学习率策略，但对于复杂的策略可能不足够。在 MMAction2 中，我们提供 `param_scheduler` 来更好地控制不同参数的学习率策略。

### 配置学习率调整策略

调整学习率策略被广泛用于提高性能。我们支持大多数 PyTorch 学习率策略，包括 `ExponentialLR`、`LinearLR`、`StepLR`、`MultiStepLR` 等。

所有可用的学习率策略可以在[这里](https://mmaction2.readthedocs.io/en/latest/schedulers.html)找到，学习率策略的名称以 `LR` 结尾。

- **单一学习率策略**

  在大多数情况下，我们只使用一个学习策略以简化问题。例如，`MultiStepLR` 被用作 ResNet 的默认学习率策略。在这里，`param_scheduler` 是一个字典。

  ```python
  param_scheduler = dict(
      type='MultiStepLR',
      by_epoch=True,
      milestones=[100, 150],
      gamma=0.1)
  ```

  或者，我们想使用 `CosineAnnealingLR` 策略来衰减学习率：

  ```python
  param_scheduler = dict(
      type='CosineAnnealingLR',
      by_epoch=True,
      T_max=num_epochs)
  ```

- **多个学习率策略**

  在某些训练案例中，为了提高准确性，会应用多个学习率策略。例如，在早期阶段，训练容易不稳定，预热是一种减少不稳定性的技术。学习率将从一个较小的值逐渐增加到预期值，通过预热进行衰减和其他策略进行衰减。

  在 MMAction2 中，通过将所需的策略组合成 `param_scheduler` 的列表即可实现预热策略。

  以下是一些示例：

  1. 在前 50 个迭代中进行线性预热。

  ```python
    param_scheduler = [
        # 线性预热
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=False,  # 按迭代
            end=50),  # 仅在前 50 个迭代中进行预热
        # 主要的学习率策略
        dict(type='MultiStepLR',
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1)
    ]
  ```

  2. 在前 10 个周期中进行线性预热，并在每个周期内按迭代更新学习率。

  ```python
    param_scheduler = [
        # 线性预热 [0, 10) 个周期
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=True,
            end=10,
            convert_to_iter_based=True,  # 按迭代更新学习率
        ),
        # 在 10 个周期后使用 CosineAnnealing 策略
        dict(type='CosineAnnealingLR', by_epoch=True, begin=10)
    ]
  ```

  注意，我们在这里使用 `begin` 和 `end` 参数来指定有效范围，该范围为 \[`begin`, `end`)。范围的单位由 `by_epoch` 参数定义。如果未指定，则 `begin` 为 0，`end` 为最大周期或迭代次数。

  如果所有策略的范围都不连续，则学习率将在忽略的范围内保持不变，否则所有有效的策略将按特定阶段的顺序执行，这与 PyTorch [`ChainedScheduler`](torch.optim.lr_scheduler.ChainedScheduler) 的行为相同。

### 自定义动量策略

我们支持使用动量策略根据学习率修改优化器的动量，这可以使损失以更快的方式收敛。使用方法与学习率策略相同。

所有可用的学习率策略可以在[这里](https://mmaction2.readthedocs.io/en/latest/schedulers.html)找到，动量策略的名称以 `Momentum` 结尾。

以下是一个示例：

```python
param_scheduler = [
    # 学习率策略
    dict(type='LinearLR', ...),
    # 动量策略
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## 添加新的优化器或构造器

本部分将修改 MMAction2 源代码或向 MMAction2 框架中添加代码，初学者可以跳过此部分。

### 添加新的优化器

在学术研究和工业实践中，可能需要使用 MMAction2 未实现的优化方法，可以通过以下方法进行添加。

#### 1. 实现一个新的优化器

假设要添加一个名为 `MyOptimizer` 的优化器，它具有参数 `a`、`b` 和 `c`。需要在 `mmaction/engine/optimizers` 下创建一个新文件，并在文件中实现新的优化器，例如在 `mmaction/engine/optimizers/my_optimizer.py` 中：

```python
from torch.optim import Optimizer
from mmaction.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
        ...

    def step(self, closure=None):
        ...
```

#### 2. 导入优化器

为了找到上述定义的模块，需要在运行时导入该模块。首先，在 `mmaction/engine/optimizers/__init__.py` 中导入该模块，将其添加到 `mmaction.engine` 包中。

```python
# In mmaction/engine/optimizers/__init__.py
...
from .my_optimizer import MyOptimizer # MyOptimizer 可能是其他类名

__all__ = [..., 'MyOptimizer']
```

在运行时，我们将自动导入 `mmaction.engine` 包，并同时注册 `MyOptimizer`。

#### 3. 在配置文件中指定优化器

然后，可以在配置文件的 `optim_wrapper.optimizer` 字段中使用 `MyOptimizer`。

```python
optim_wrapper = dict(
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### 添加新的优化器构造器

一些模型可能对优化有一些特定的参数设置，例如所有 `BatchNorm` 层的不同权重衰减率。

尽管我们已经可以使用[优化器教程](#参数化精细配置)中的 `optim_wrapper.paramwise_cfg` 字段来配置各种特定参数的优化器设置，但可能仍无法满足需求。

当然，你可以修改它。默认情况下，我们使用 [`DefaultOptimWrapperConstructor`](mmengine.optim.DefaultOptimWrapperConstructor) 类来处理优化器的构造。在构造过程中，它根据 `paramwise_cfg` 对不同参数的优化器设置进行细致配置，这也可以作为新优化器构造器的模板。

你可以通过添加新的优化器构造器来覆盖这些行为。

```python
# In mmaction/engine/optimizers/my_optim_constructor.py
from mmengine.optim import DefaultOptimWrapperConstructor
from mmaction.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class MyOptimWrapperConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        ...

    def __call__(self, model):
        ...
```

然后，导入它并几乎像[优化器教程](#添加新的优化器)中那样使用它。

1. 在 `mmaction/engine/optimizers/__init__.py` 中导入它，将其添加到 `mmaction.engine` 包中。

   ```python
   # In mmaction/engine/optimizers/__init__.py
   ...
   from .my_optim_constructor import MyOptimWrapperConstructor

   __all__ = [..., 'MyOptimWrapperConstructor']
   ```

2. 在配置文件的 `optim_wrapper.constructor` 字段中使用 `MyOptimWrapperConstructor`。

   ```python
   optim_wrapper = dict(
       constructor=dict(type='MyOptimWrapperConstructor'),
       optimizer=...,
       paramwise_cfg=...,
   )
   ```
