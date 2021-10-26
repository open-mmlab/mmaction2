# 教程 5：如何添加新模块

在本教程中，我们将介绍一些有关如何为该项目定制优化器，开发新组件，以及添加新的学习率调整器（更新器）的方法。

<!-- TOC -->

- [自定义优化器](#自定义优化器)
- [自定义优化器构造器](#自定义优化器构造器)
- [开发新组件](#开发新组件)
  - [添加新的 backbones](#添加新-backbones)
  - [添加新的 heads](#添加新-heads)
  - [添加新的 loss function](#添加新-loss-function)
- [添加新的学习率调节器（更新器）](#添加新的学习率调节器（更新器）)

<!-- TOC -->

## 自定义优化器

[CopyOfSGD](/mmaction/core/optimizer/copy_of_sgd.py) 是自定义优化器的一个例子，写在 `mmaction/core/optimizer/copy_of_sgd.py` 文件中。
更一般地，可以根据如下方法自定义优化器。

假设添加的优化器名为 `MyOptimizer`，它有 `a`，`b` 和 `c` 三个参数。
用户需要首先实现一个新的优化器文件，如 `mmaction/core/optimizer/my_optimizer.py`：

```python
from mmcv.runner import OPTIMIZERS
from torch.optim import Optimizer

@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
```

然后添加这个模块到 `mmaction/core/optimizer/__init__.py` 中，从而让注册器可以找到这个新的模块并添加它：

```python
from .my_optimizer import MyOptimizer
```

之后，用户便可以在配置文件的 `optimizer` 字段中使用 `MyOptimizer`。
在配置中，优化器由 `optimizer` 字段所定义，如下所示：

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

用户可以直接根据 [PyTorch API 文档](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) 对参数进行直接设置。

## 自定义优化器构造器

某些模型可能对不同层的参数有特定的优化设置，例如 BatchNorm 层的梯度衰减。
用户可以通过自定义优化器构造函数来进行那些细粒度的参数调整。

用户可以编写一个基于 [DefaultOptimizerConstructor](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py) 的新的优化器构造器，
并且重写 `add_params(self, params, module)` 方法。

一个自定义优化器构造器的例子是 [TSMOptimizerConstructor](/mmaction/core/optimizer/tsm_optimizer_constructor.py)。
更具体地，可以如下定义定制的优化器构造器。

在 `mmaction/core/optimizer/my_optimizer_constructor.py`：

```python
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor

@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):

```

在 `mmaction/core/optimizer/__init__.py`：

```python
from .my_optimizer_constructor import MyOptimizerConstructor
```

之后便可在配置文件的 `optimizer` 域中使用 `MyOptimizerConstructor`。

```python
# 优化器
optimizer = dict(
    type='SGD',
    constructor='MyOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001)
```

## 开发新组件

MMAction2 将模型组件分为 4 种基础模型：

- 识别器（recognizer）：整个识别器模型流水线，通常包含一个主干网络（backbone）和分类头（cls_head）。
- 主干网络（backbone）：通常为一个用于提取特征的 FCN 网络，例如 ResNet，BNInception。
- 分类头（cls_head）：用于分类任务的组件，通常包括一个带有池化层的 FC 层。
- 时序检测器（localizer）：用于时序检测的模型，目前有的检测器包含 BSN，BMN，SSN。

### 添加新的 backbones

这里以 TSN 为例，说明如何开发新的组件。

1. 创建新文件 `mmaction/models/backbones/resnet.py`

    ```python
    import torch.nn as nn

    from ..builder import BACKBONES

    @BACKBONES.register_module()
    class ResNet(nn.Module):

        def __init__(self, arg1, arg2):
            pass

        def forward(self, x):  # 应该返回一个元组
            pass

        def init_weights(self, pretrained=None):
            pass
    ```

2. 在 `mmaction/models/backbones/__init__.py` 中导入模型

    ```python
    from .resnet import ResNet
    ```

3. 在配置文件中使用它

    ```python
    model = dict(
        ...
        backbone=dict(
            type='ResNet',
            arg1=xxx,
            arg2=xxx),
    )
    ```

### 添加新的 heads

这里以 TSNHead 为例，说明如何开发新的 head

1. 创建新文件 `mmaction/models/heads/tsn_head.py`

    可以通过继承 [BaseHead](/mmaction/models/heads/base.py) 编写一个新的分类头，
    并重写 `init_weights(self)` 和 `forward(self, x)` 方法

    ```python
    from ..builder import HEADS
    from .base import BaseHead


    @HEADS.register_module()
    class TSNHead(BaseHead):

        def __init__(self, arg1, arg2):
            pass

        def forward(self, x):
            pass

        def init_weights(self):
            pass
    ```

2. 在 `mmaction/models/heads/__init__.py` 中导入模型

    ```python
    from .tsn_head import TSNHead
    ```

3. 在配置文件中使用它

    ```python
    model = dict(
        ...
        cls_head=dict(
            type='TSNHead',
            num_classes=400,
            in_channels=2048,
            arg1=xxx,
            arg2=xxx),
    ```

### 添加新的 loss function

假设用户想添加新的 loss 为 `MyLoss`。为了添加一个新的损失函数，需要在 `mmaction/models/losses/my_loss.py` 下进行实现。

```python
import torch
import torch.nn as nn

from ..builder import LOSSES

def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


@LOSSES.register_module()
class MyLoss(nn.Module):

    def forward(self, pred, target):
        loss = my_loss(pred, target)
        return loss
```

之后，用户需要把它添加进 `mmaction/models/losses/__init__.py`

```python
from .my_loss import MyLoss, my_loss
```

为了使用它，需要修改 `loss_xxx` 域。由于 MyLoss 用户识别任务，可以把它作为边界框损失 `loss_bbox`

```python
loss_bbox=dict(type='MyLoss'))
```

### 添加新的学习率调节器（更新器）

构造学习率更新器（即 PyTorch 中的 "scheduler"）的默认方法是修改配置，例如：

```python
...
lr_config = dict(policy='step', step=[20, 40])
...
```

在 [`train.py`](/mmaction/apis/train.py) 的 api 中，它会在以下位置注册用于学习率更新的钩子：

```python
...
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None))
...
```

到目前位置，所有支持的更新器可参考 [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)，
但如果用户想自定义学习率更新器，则需要遵循以下步骤：

1. 首先，在 `$MMAction2/mmaction/core/scheduler` 编写自定义的学习率更新钩子（LrUpdaterHook）。以下片段是自定义学习率更新器的例子，它使用基于特定比率的学习率 `lrs`，并在每个 `steps` 处进行学习率衰减。以下代码段是自定义学习率更新器的例子：

```python
# 在此注册
@HOOKS.register_module()
class RelativeStepLrUpdaterHook(LrUpdaterHook):
    # 该类应当继承于 mmcv.LrUpdaterHook
    def __init__(self, steps, lrs, **kwargs):
        super().__init__(**kwargs)
        assert len(steps) == (len(lrs))
        self.steps = steps
        self.lrs = lrs

    def get_lr(self, runner, base_lr):
        # 仅需要重写该函数
        # 该函数在每个训练周期之前被调用, 并返回特定的学习率.
        progress = runner.epoch if self.by_epoch else runner.iter
        for i in range(len(self.steps)):
            if progress < self.steps[i]:
                return self.lrs[i]
```

2. 修改配置

在配置文件下替换原先的 `lr_config` 变量

```python
lr_config = dict(policy='RelativeStep', steps=[20, 40, 60], lrs=[0.1, 0.01, 0.001])
```

更多例子可参考 [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)
