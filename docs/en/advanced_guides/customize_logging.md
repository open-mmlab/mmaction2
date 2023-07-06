# Customize Logging

MMAction2 produces a lot of logs during the running process, such as loss, iteration time, learning rate, etc. In this section, we will introduce you how to output custom log. More details about the logging system, please refer to [MMEngine Tutorial](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/logging.html).

- [Customize Logging](#customize-logging)
  - [Flexible Logging System](#flexible-logging-system)
  - [Customize log](#customize-log)
  - [Export the debug log](#export-the-debug-log)

## Flexible Logging System

The MMAction2 logging system is configured by the `LogProcessor` in [default_runtime](https://github.com/open-mmlab/mmaction2/tree/main/configs/_base_/default_runtime.py) by default, which is equivalent to:

```python
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
```

By default, the `LogProcessor` captures all fields that begin with `loss` returned by `model.forward`. For instance, in the following model, `loss1` and `loss2` will be logged automatically without any additional configuration.

```python
from mmengine.model import BaseModel

class ToyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss1 = (feat - label).pow(2)
        loss2 = (feat - label).abs()
        return dict(loss1=loss1, loss2=loss2)
```

The output log follows the following format:

```
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0019  data_time: 0.0004  loss1: 0.8381  loss2: 0.9007  loss: 1.7388
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0029  data_time: 0.0010  loss1: 0.1978  loss2: 0.4312  loss: 0.6290
```

`LogProcessor` will output the log in the following format:

- The prefix of the log:
  - epoch mode(`by_epoch=True`): `Epoch(train) [{current_epoch}/{current_iteration}]/{dataloader_length}`
  - iteration mode(`by_epoch=False`): `Iter(train) [{current_iteration}/{max_iteration}]`
- Learning rate (`lr`): The learning rate of the last iteration.
- Time:
  - `time`: The averaged time for inference of the last `window_size` iterations.
  - `data_time`: The averaged time for loading data of the last `window_size` iterations.
  - `eta`: The estimated time of arrival to finish the training.
- Loss: The averaged loss output by model of the last `window_size` iterations.

```{warning}
log_processor outputs the epoch based log by default(`by_epoch=True`). To get an expected log matched with the `train_cfg`, we should set the same value for `by_epoch` in `train_cfg` and `log_processor`.
```

Based on the rules above, the code snippet will count the average value of the loss1 and the loss2 every 20 iterations. More types of statistical methods, please refer to [mmengine.runner.LogProcessor](mmengine.runner.LogProcessor).

## Customize log

The logging system could not only log the `loss`, `lr`, .etc but also collect and output the custom log. For example, if we want to statistic the intermediate loss:

The `ToyModel` calculate `loss_tmp` in forward, but don't save it into the return dict.

```python
from mmengine.logging import MessageHub

class ToyModel(BaseModel):

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, img, label, mode):
        feat = self.linear(img)
        loss_tmp = (feat - label).abs()
        loss = loss_tmp.pow(2)

        message_hub = MessageHub.get_current_instance()
        # update the intermediate `loss_tmp` in the message hub
        message_hub.update_scalar('train/loss_tmp', loss_tmp.sum())
        return dict(loss=loss)
```

Add the `loss_tmp` into the config:

```python
log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True,
    custom_cfg=[
        # statistic the loss_tmp with the averaged value
            dict(
                data_src='loss_tmp',
                window_size=20,
                method_name='mean')
        ])
```

The `loss_tmp` will be added to the output log:

```
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0008  loss_tmp: 0.0097  loss: 0.0000
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0028  data_time: 0.0013  loss_tmp: 0.0065  loss: 0.0000
```

## Export the debug log

To export the debug log to the `work_dir`, you can set log_level in config file as follows:

```
log_level='DEBUG'
```

```
08/21 18:16:22 - mmengine - DEBUG - Get class `LocalVisBackend` from "vis_backend" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `LocalVisBackend` instance is built from registry, its implementation can be found in mmengine.visualization.vis_backend
08/21 18:16:22 - mmengine - DEBUG - Get class `RuntimeInfoHook` from "hook" registry in "mmengine"
08/21 18:16:22 - mmengine - DEBUG - An `RuntimeInfoHook` instance is built from registry, its implementation can be found in mmengine.hooks.runtime_info_hook
08/21 18:16:22 - mmengine - DEBUG - Get class `IterTimerHook` from "hook" registry in "mmengine"
...
```

Besides, logs of different ranks will be saved in `debug` mode if you are training your model with the shared storage. The hierarchy of the log is as follows:

```text
./tmp
├── tmp.log
├── tmp_rank1.log
├── tmp_rank2.log
├── tmp_rank3.log
├── tmp_rank4.log
├── tmp_rank5.log
├── tmp_rank6.log
└── tmp_rank7.log
...
└── tmp_rank63.log
```

The log of Multiple machines with independent storage:

```text
# device: 0:
work_dir/
└── exp_name_logs
    ├── exp_name.log
    ├── exp_name_rank1.log
    ├── exp_name_rank2.log
    ├── exp_name_rank3.log
    ...
    └── exp_name_rank7.log

# device: 7:
work_dir/
└── exp_name_logs
    ├── exp_name_rank56.log
    ├── exp_name_rank57.log
    ├── exp_name_rank58.log
    ...
    └── exp_name_rank63.log
```
