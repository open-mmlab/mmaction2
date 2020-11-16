# Tutorial 4: Adding New Modules

In this tutorial, we will introduce some methods about how to customize optimizer, develop new components and new a learning rate scheduler for this project.

<!-- TOC -->

- [Customize Optimizer](#customize-optimizer)
- [Customize Optimizer Constructor](#customize-optimizer-constructor)
- [Develop New Components](#develop-new-components)
  * [Add new backbones](#add-new-backbones)
  * [Add new heads](#add-new-heads)
  * [Add new loss](#add-new-loss)
- [Add new learning rate scheduler (updater)](#add-new-learning-rate-scheduler--updater-)

<!-- TOC -->

## Customize Optimizer

An example of customized optimizer is [CopyOfSGD](/mmaction/core/optimizer/copy_of_sgd.py) is defined in `mmaction/core/optimizer/copy_of_sgd.py`.
More generally, a customized optimizer could be defined as following.

Assume you want to add an optimizer named as `MyOptimizer`, which has arguments `a`, `b` and `c`.
You need to first implement the new optimizer in a file, e.g., in `mmaction/core/optimizer/my_optimizer.py`:

```python
from .registry import OPTIMIZERS
from torch.optim import Optimizer

@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c):
```

Then add this module in `mmaction/core/optimizer/__init__.py`, thus the registry will find the new module and add it:

```python
from .my_optimizer import MyOptimizer
```

Then you can use `MyOptimizer` in `optimizer` field of config files.
In the configs, the optimizers are defined by the field `optimizer` like the following:

```python
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
```

To use your own optimizer, the field can be changed as

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
```

We already support to use all the optimizers implemented by PyTorch, and the only modification is to change the `optimizer` field of config files.
For example, if you want to use `ADAM`, though the performance will drop a lot, the modification could be as the following.

```python
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

The users can directly set arguments following the [API doc](https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim) of PyTorch.

## Customize Optimizer Constructor

Some models may have some parameter-specific settings for optimization, e.g. weight decay for BatchNorm layers.
The users can do those fine-grained parameter tuning through customizing optimizer constructor.

You can write a new optimizer constructor inherit from [DefaultOptimizerConstructor](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/optimizer/default_constructor.py)
and overwrite the `add_params(self, params, module)` method.

An example of customized optimizer constructor is [TSMOptimizerConstructor](/mmaction/core/optimizer/tsm_optimizer_constructor.py).
More generally, a customized optimizer constructor could be defined as following.

In `mmaction/core/optimizer/my_optimizer_constructor.py`:

```python
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor

@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):

```

In `mmaction/core/optimizer/__init__.py`:

```python
from .my_optimizer_constructor import MyOptimizerConstructor
```

Then you can use `MyOptimizerConstructor` in `optimizer` field of config files.

```python
# optimizer
optimizer = dict(
    type='SGD',
    constructor='MyOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001)
```

## Develop New Components

We basically categorize model components into 4 types.

- recognizer: the whole recognizer model pipeline, usually contains a backbone and cls_head.
- backbone: usually an FCN network to extract feature maps, e.g., ResNet, BNInception.
- cls_head: the component for classification task, usually contains an FC layer with some pooling layers.
- localizer: the model for temporal localization task, currently available: BSN, BMN.

### Add new backbones

Here we show how to develop new components with an example of TSN.

1. Create a new file `mmaction/models/backbones/resnet.py`.

    ```python
    import torch.nn as nn

    from ..registry import BACKBONES

    @BACKBONES.register_module()
    class ResNet(nn.Module):

        def __init__(self, arg1, arg2):
            pass

        def forward(self, x):  # should return a tuple
            pass

        def init_weights(self, pretrained=None):
            pass
    ```

2. Import the module in `mmaction/models/backbones/__init__.py`.

    ```python
    from .resnet import ResNet
    ```

3. Use it in your config file.

    ```python
    model = dict(
        ...
        backbone=dict(
            type='ResNet',
            arg1=xxx,
            arg2=xxx),
    )
    ```

### Add new heads

Here we show how to develop a new head with the example of TSNHead as the following.

1. Create a new file `mmaction/models/heads/tsn_head.py`.

    You can write a new classification head inheriting from [BaseHead](/mmaction/models/heads/base.py),
    and overwrite `init_weights(self)` and `forward(self, x)` method.

    ```python
    from ..registry import HEADS
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

2. Import the module in `mmaction/models/heads/__init__.py`

    ```python
    from .tsn_head import TSNHead
    ```

3. Use it in your config file

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

### Add new loss

Assume you want to add a new loss as `MyLoss`. To add a new loss function, the users need implement it in `mmaction/models/losses/my_loss.py`.

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

Then the users need to add it in the `mmaction/models/losses/__init__.py`

```python
from .my_loss import MyLoss, my_loss
```

To use it, modify the `loss_xxx` field. Since MyLoss is for regression, we can use it for the bbox loss `loss_bbox`.

```python
loss_bbox=dict(type='MyLoss'))
```

## Add new learning rate scheduler (updater)
The default manner of constructing a lr updater(namely, 'scheduler' by pytorch convention), is to modify the config such as:
```python
...
lr_config = dict(policy='step', step=[20, 40])
...
```
In the api for [`train.py`](/mmaction/apis/train.py), it will register the learning rate updater hook based on the config at:
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
So far, the supported updaters can be find in [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py), but if you want to customize a new learning rate updater, you may follow the steps below:

1. First, write your own LrUpdaterHook in `$MMAction2/mmaction/core/lr`. The snippet followed is an example of cumtomized lr updater that uses learning rate based on a specific learning rate ratio: `lrs`, by which the learning rate decreases at each `steps`:
```python
@HOOKS.register_module()
# Register it here
class RelativeStepLrUpdaterHook(LrUpdaterHook):
    # You should inheritate it from mmcv.LrUpdaterHook
    def __init__(self, runner, steps, lrs, **kwargs):
        super().__init__(**kwargs)
        assert len(steps) == (len(lrs))
        self.steps = steps
        self.lrs = lrs

    def get_lr(self, runner, base_lr):
        # Only this function is required to override
        # This function is called before each training epoch, return the specific learning rate here.
        progress = runner.epoch if self.by_epoch else runner.iter
        for i in range(len(self.steps)):
            if progress < self.steps[i]:
                return self.lrs[i]
```

2. Modify your config:
In your config file, swap the original `lr_config` by:
```python
lr_config = dict(policy='RelativeStep', steps=[20, 40, 60], lrs=[0.1, 0.01, 0.001])
```

More examples can be found in [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py).
