# 自定义日志

MMAction2 在运行过程中会产生大量的日志，如损失、迭代时间、学习率等。在这一部分，我们将向你介绍如何输出自定义日志。有关日志系统的更多详细信息，请参考 [MMEngine 教程](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/logging.html)。

- [自定义日志](#自定义日志)
  - [灵活的日志系统](#灵活的日志系统)
  - [定制日志](#定制日志)
  - [导出调试日志](#导出调试日志)

## 灵活的日志系统

默认情况下，MMAction2 的日志系统由 [default_runtime](/configs/_base_/default_runtime.py) 中的 `LogProcessor` 配置：

```python
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
```

默认情况下，`LogProcessor` 捕获 `model.forward` 返回的所有以 `loss` 开头的字段。例如，在以下模型中，`loss1` 和 `loss2` 将在没有任何额外配置的情况下自动记录到日志。

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

输出日志遵循以下格式：

```
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0019  data_time: 0.0004  loss1: 0.8381  loss2: 0.9007  loss: 1.7388
08/21 02:58:41 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0029  data_time: 0.0010  loss1: 0.1978  loss2: 0.4312  loss: 0.6290
```

`LogProcessor` 将按以下格式输出日志：

- 日志的前缀：
  - epoch 模式(`by_epoch=True`)：`Epoch(train) [{current_epoch}/{current_iteration}]/{dataloader_length}`
  - iteration 模式(`by_epoch=False`)：`Iter(train) [{current_iteration}/{max_iteration}]`
- 学习率 (`lr`)：最后一次迭代的学习率。
- 时间：
  - `time`：过去 `window_size` 次迭代的推理平均时间。
  - `data_time`：过去 `window_size` 次迭代的数据加载平均时间。
  - `eta`：完成训练的预计到达时间。
- 损失：过去 `window_size` 次迭代中模型输出的平均损失。

```{warning}
默认情况下，log_processor 输出基于 epoch 的日志(`by_epoch=True`)。要得到与 `train_cfg` 匹配的预期日志，我们应在 `train_cfg` 和 `log_processor` 中设置相同的 `by_epoch` 值。
```

根据以上规则，代码片段将每20次迭代计算 loss1 和 loss2 的平均值。更多类型的统计方法，请参考 [mmengine.runner.LogProcessor](mmengine.runner.LogProcessor)。

## 定制日志

日志系统不仅可以记录 `loss`，`lr` 等，还可以收集和输出自定义日志。例如，如果我们想要统计中间损失：

`ToyModel` 在 forward 中计算 `loss_tmp`，但不将其保存到返回字典中。

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
        # 在消息中心更新中间的 `loss_tmp`
        message_hub.update_scalar('train/loss_tmp', loss_tmp.sum())
        return dict(loss=loss)
```

将 `loss_tmp` 添加到配置中：

```python
log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True,
    custom_cfg=[
        # 使用平均值统计 loss_tmp
            dict(
                data_src='loss_tmp',
                window_size=20,
                method_name='mean')
        ])
```

`loss_tmp` 将被添加到输出日志中：

```
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][10/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0026  data_time: 0.0008  loss_tmp: 0.0097  loss: 0.0000
08/21 03:40:31 - mmengine - INFO - Epoch(train) [1][20/25]  lr: 1.0000e-02  eta: 0:00:00  time: 0.0028  data_time: 0.0013  loss_tmp: 0.0065  loss: 0.0000
```

## 导出调试日志

要将调试日志导出到 `work_dir`，你可以在配置文件中设置日志级别如下：

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

此外，如果你正在使用共享存储训练你的模型，那么在 `debug` 模式下，不同排名的日志将被保存。日志的层级结构如下：

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

在具有独立存储的多台机器上的日志：

```text
# 设备：0：
work_dir/
└── exp_name_logs
    ├── exp_name.log
    ├── exp_name_rank1.log
    ├── exp_name_rank2.log
    ├── exp_name_rank3.log
    ...
    └── exp_name_rank7.log

# 设备：7：
work_dir/
└── exp_name_logs
    ├── exp_name_rank56.log
    ├── exp_name_rank57.log
    ├── exp_name_rank58.log
    ...
    └── exp_name_rank63.log
```
