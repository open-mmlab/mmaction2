# Customize Optimizer

In this tutorial, we will introduce some methods about how to build the optimizer and learning rate scheduler for your tasks.

- [Customize Optimizer](#customize-optimizer)
  - [Build optimizers using optim_wrapper](#build-optimizers-using-optim_wrapper)
    - [Use optimizers supported by PyTorch](#use-optimizers-supported-by-pytorch)
    - [Parameter-wise finely configuration](#parameter-wise-finely-configuration)
    - [Gradient clipping](#gradient-clipping)
    - [Gradient accumulation](#gradient-accumulation)
  - [Customize parameter schedules](#customize-parameter-schedules)
    - [Customize learning rate schedules](#customize-learning-rate-schedules)
    - [Customize momentum schedules](#customize-momentum-schedules)
  - [Add new optimizers or constructors](#add-new-optimizers-or-constructors)
    - [Add new optimizers](#add-new-optimizers)
      - [1. Implement a new optimizer](#1-implement-a-new-optimizer)
      - [2. Import the optimizer](#2-import-the-optimizer)
      - [3. Specify the optimizer in the config file](#3-specify-the-optimizer-in-the-config-file)
    - [Add new optimizer constructors](#add-new-optimizer-constructors)

## Build optimizers using optim_wrapper

We use the `optim_wrapper` field to configure the strategies of optimization, which includes choices of the optimizer, parameter-wise configurations, gradient clipping and accumulation. A simple example can be:

```python
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0003, weight_decay=0.0001)
)
```

In the above example, a SGD optimizer with learning rate 0.0003 and weight decay 0.0001 is built.

### Use optimizers supported by PyTorch

We support all the optimizers implemented by PyTorch. To use a different optimizer, just need to change the `optimizer` field of config files. For example, if you want to use `torch.optim.Adam`, the modification in the config file could be as the following.

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

First we need to change the value of `type` to the desired optimizer name supported in `torch.optim`. Next we add necessary arguments of this optimizer to the `optimizer` field. The above config will build the following optimizer:

```python
torch.optim.Adam(lr=0.001,
                 betas=(0.9, 0.999),
                 eps=1e-08,
                 weight_decay=0,
                 amsgrad=False)
```

### Parameter-wise finely configuration

Some models may have parameter-specific settings for optimization, for example, no weight decay to the BatchNorm layers or using different learning rates for different network layers.
To finely configure them, we can use the `paramwise_cfg` argument in `optim_wrapper`.

- **Set different hyper-parameter multipliers for different types of parameters.**

  For instance, we can set `norm_decay_mult=0.` in `paramwise_cfg` to change the weight decay of weight and bias of normalization layers to zero.

  ```python
  optim_wrapper = dict(
      optimizer=dict(type='SGD', lr=0.8, weight_decay=1e-4),
      paramwise_cfg=dict(norm_decay_mult=0.))
  ```

  More types of parameters are supported to configured, list as follow:

  - `lr_mult`: Multiplier for learning rate of all parameters.
  - `decay_mult`: Multiplier for weight decay of all parameters.
  - `bias_lr_mult`: Multiplier for learning rate of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `bias_decay_mult`: Multiplier for weight decay of bias (Not include normalization layers' biases and deformable convolution layers' offsets). Defaults to 1.
  - `norm_decay_mult`: Multiplier for weight decay of weigh and bias of normalization layers. Defaults to 1.
  - `dwconv_decay_mult`: Multiplier for weight decay of depth-wise convolution layers. Defaults to 1.
  - `bypass_duplicate`: Whether to bypass duplicated parameters. Defaults to `False`.
  - `dcn_offset_lr_mult`: Multiplier for learning rate of deformable convolution layers. Defaults to 1.

- **Set different hyper-parameter multipliers for specific parameters.**

  MMAction2 can use `custom_keys` in `paramwise_cfg` to specify different parameters to use different learning rates or weight decay.

  For example, to set all learning rates and weight decays of `backbone.layer0` to 0, the rest of `backbone` remains the same as the optimizer and the learning rate of `head` to 0.001, use the configs below.

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

### Gradient clipping

During the training process, the loss function may get close to a cliffy region and cause gradient explosion. And gradient clipping is helpful to stabilize the training process. More introduction can be found in [this page](https://paperswithcode.com/method/gradient-clipping).

Currently we support `clip_grad` option in `optim_wrapper` for gradient clipping, refers to [PyTorch Documentation](torch.nn.utils.clip_grad_norm_).

Here is an example:

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    # norm_type: type of the used p-norm, here norm_type is 2.
    clip_grad=dict(max_norm=35, norm_type=2))
```

### Gradient accumulation

When computing resources are lacking, the batch size can only be set to a small value, which may affect the performance of models. Gradient accumulation can be used to solve this problem. We support `accumulative_counts` option in `optim_wrapper` for gradient accumulation.

Here is an example:

```python
train_dataloader = dict(batch_size=64)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    accumulative_counts=4)
```

Indicates that during training, back-propagation is performed every 4 iters. And the above is equivalent to:

```python
train_dataloader = dict(batch_size=256)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001))
```

## Customize parameter schedules

In training, the optimzation parameters such as learing rate, momentum, are usually not fixed but changing through iterations or epochs. PyTorch supports several learning rate schedulers, which are not sufficient for complex strategies. In MMAction2, we provide `param_scheduler` for better controls of different parameter schedules.

### Customize learning rate schedules

Learning rate schedulers are widely used to improve performance. We support most of the PyTorch schedulers, including `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc.

All available learning rate scheduler can be found {external+mmengine:ref}`here <scheduler>`, and the
names of learning rate schedulers end with `LR`.

- **Single learning rate schedule**

  In most cases, we use only one learning rate schedule for simplicity. For instance, [`MultiStepLR`](mmengine.optim.MultiStepLR) is used as the default learning rate schedule for ResNet. Here, `param_scheduler` is a dictionary.

  ```python
  param_scheduler = dict(
      type='MultiStepLR',
      by_epoch=True,
      milestones=[100, 150],
      gamma=0.1)
  ```

  Or, we want to use the [`CosineAnnealingLR`](mmengine.optim.CosineAnnealingLR) scheduler to decay the learning rate:

  ```python
  param_scheduler = dict(
      type='CosineAnnealingLR',
      by_epoch=True,
      T_max=num_epochs)
  ```

- **Multiple learning rate schedules**

  In some of the training cases, multiple learning rate schedules are applied for higher accuracy. For example ,in the early stage, training is easy to be volatile, and warmup is a technique to reduce volatility.
  The learning rate will increase gradually from a minor value to the expected value by warmup and decay afterwards by other schedules.

  In MMAction2, simply combines desired schedules in `param_scheduler` as a list can achieve the warmup strategy.

  Here are some examples:

  1. linear warmup during the first 50 iters.

  ```python
    param_scheduler = [
        # linear warm-up by iters
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=False,  # by iters
            end=50),  # only warm up for first 50 iters
        # main learing rate schedule
        dict(type='MultiStepLR',
            by_epoch=True,
            milestones=[8, 11],
            gamma=0.1)
    ]
  ```

  2. linear warmup and update lr by iter during the first 10 epochs.

  ```python
    param_scheduler = [
        # linear warm-up by epochs in [0, 10) epochs
        dict(type='LinearLR',
            start_factor=0.001,
            by_epoch=True,
            end=10,
            convert_to_iter_based=True,  # Update learning rate by iter.
        ),
        # use CosineAnnealing schedule after 10 epochs
        dict(type='CosineAnnealingLR', by_epoch=True, begin=10)
    ]
  ```

  Notice that, we use `begin` and `end` arguments here to assign the valid range, which is \[`begin`, `end`) for this schedule. And the range unit is defined by `by_epoch` argument. If not specified, the `begin` is 0 and the `end` is the max epochs or iterations.

  If the ranges for all schedules are not continuous, the learning rate will stay constant in ignored range, otherwise all valid schedulers will be executed in order in a specific stage, which behaves the same as PyTorch [`ChainedScheduler`](torch.optim.lr_scheduler.ChainedScheduler).

### Customize momentum schedules

We support using momentum schedulers to modify the optimizer's momentum according to learning rate, which could make the loss converge in a faster way. The usage is the same as learning rate schedulers.

All available learning rate scheduler can be found {external+mmengine:ref}`here <scheduler>`, and the
names of momentum rate schedulers end with `Momentum`.

Here is an example:

```python
param_scheduler = [
    # the lr scheduler
    dict(type='LinearLR', ...),
    # the momentum scheduler
    dict(type='LinearMomentum',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=1000)
]
```

## Add new optimizers or constructors

This part will modify the MMAction2 source code or add code to the MMAction2 framework, beginners can skip it.

### Add new optimizers

In academic research and industrial practice, it may be necessary to use optimization methods not implemented by MMAction2, and you can add them through the following methods.

#### 1. Implement a new optimizer

Assume you want to add an optimizer named `MyOptimizer`, which has arguments `a`, `b`, and `c`.
You need to create a new file under `mmaction/engine/optimizers`, and implement the new optimizer in the file, for example, in `mmaction/engine/optimizers/my_optimizer.py`:

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

#### 2. Import the optimizer

To find the above module defined above, this module should be imported during the running. First import it in the `mmaction/engine/optimizers/__init__.py` to add it into the `mmaction.engine` package.

```python
# In mmaction/engine/optimizers/__init__.py
...
from .my_optimizer import MyOptimizer # MyOptimizer maybe other class name

__all__ = [..., 'MyOptimizer']
```

During running, we will automatically import the `mmaction.engine` package and register the `MyOptimizer` at the same time.

#### 3. Specify the optimizer in the config file

Then you can use `MyOptimizer` in the `optim_wrapper.optimizer` field of config files.

```python
optim_wrapper = dict(
    optimizer=dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value))
```

### Add new optimizer constructors

Some models may have some parameter-specific settings for optimization, like different weight decay rate for all `BatchNorm` layers.

Although we already can use [the `optim_wrapper.paramwise_cfg` field](#parameter-wise-finely-configuration) to
configure various parameter-specific optimizer settings. It may still not cover your need.

Of course, you can modify it. By default, we use the [`DefaultOptimWrapperConstructor`](mmengine.optim.DefaultOptimWrapperConstructor)
class to deal with the construction of optimizer. And during the construction, it fine-grainedly configures the optimizer settings of
different parameters according to the `paramwise_cfg`，which could also serve as a template for new optimizer constructor.

You can overwrite these behaviors by add new optimizer constructors.

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

And then, import it and use it almost like [the optimizer tutorial](#add-new-optimizers).

1. Import it in the `mmaction/engine/optimizers/__init__.py` to add it into the `mmaction.engine` package.

   ```python
   # In mmaction/engine/optimizers/__init__.py
   ...
   from .my_optim_constructor import MyOptimWrapperConstructor

   __all__ = [..., 'MyOptimWrapperConstructor']
   ```

2. Use `MyOptimWrapperConstructor` in the `optim_wrapper.constructor` field of config files.

   ```python
   optim_wrapper = dict(
       constructor=dict(type='MyOptimWrapperConstructor'),
       optimizer=...,
       paramwise_cfg=...,
   )
   ```
