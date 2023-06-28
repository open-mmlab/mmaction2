# 快速运行

本章将介绍 MMAction2 的基本功能。我们假设你已经[源码安装 MMAction2](installation.md#best-practices)。

- [快速运行](#快速运行)
  - [推理](#推理)
  - [准备数据集](#准备数据集)
  - [修改配置](#修改配置)
    - [修改数据集](#修改数据集)
    - [修改运行配置](#修改运行配置)
    - [修改模型配置](#修改模型配置)
  - [浏览数据集](#浏览数据集)
  - [训练](#训练)
  - [测试](#测试)

## 推理

在 MMAction2 的根目录下执行如下命令:

```shell
python demo/demo_inferencer.py  demo/demo.mp4 \
    --rec tsn --print-result \
    --label-file tools/data/kinetics/label_map_k400.txt
```

您应该能够看到弹出的视频窗口，和在控制台中打印的推断结果。

<div align="center">
    <img src="https://user-images.githubusercontent.com/33249023/227216933-29b84ac7-ca0e-408d-b4d2-5a2e5a7357bf.gif" height="250"/>
</div>
<br />

```bash
# 推理结果
{'predictions': [{'rec_labels': [[6]], 'rec_scores': [[...]]}]}
```

```{note}
如果您在没有 GUI 的服务器上运行 MMAction2，或者通过禁用 X11 转发的 SSH 隧道运行 MMAction2，则可能不会看到弹出窗口。
```

关于 MMAction2 推理接口的详细描述可以在[这里](/demo/README.md#inferencer)找到.

除了使用我们提供的预训练模型，您还可以在自己的数据集上训练模型。在下一节中，我们将通过在精简版 [Kinetics](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) 数据集上训练 TSN 为例，带您了解 MMAction2 的基本功能。

## 准备数据集

由于视频数据集格式的多样性不利于数据集的切换，MMAction2 提出了统一的[数据格式](../user_guides/prepare_dataset.md) ，并为常用的视频数据集提供了[数据集准备指南](../user_guides/data_prepare/dataset_prepare.md)。通常，要在 MMAction2 中使用这些数据集，你只需要按照步骤进行准备。

```{笔记}
但在这里，效率意味着一切。
```

首先，请下载我们预先准备好的 [kinetics400_tiny.zip](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) ，并将其解压到 MMAction2 根目录下的 `data/` 目录。这将为您提供必要的视频和注释文件。

```Bash
wget https://download.openmmlab.com/mmaction/kinetics400_tiny.zip
mkdir -p data/
unzip kinetics400_tiny.zip -d data/
```

## 修改配置

准备好数据集之后，下一步是修改配置文件，以指定训练集和训练参数的位置。

在本例中，我们将使用 resnet50 作为主干网络来训练 TSN。由于 MMAction2 已经有了完整的 Kinetics400 数据集的配置文件 (`configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`)，我们只需要在其基础上进行一些修改。

### 修改数据集

我们首先需要修改数据集的路径。打开 `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` ，按如下替换关键字:

```Python
data_root = 'data/kinetics400_tiny/train'
data_root_val = 'data/kinetics400_tiny/val'
ann_file_train = 'data/kinetics400_tiny/kinetics_tiny_train_video.txt'
ann_file_val = 'data/kinetics400_tiny/kinetics_tiny_val_video.txt'
```

### 修改运行配置

此外，由于数据集的大小减少，我们建议将训练批大小减少到4个，训练epoch的数量相应减少到10个。此外，我们建议将验证和权值存储间隔缩短为1轮，并修改学习率衰减策略。修改 `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 中对应的关键字，如下所示生效。

```python
# 设置训练批大小为 4
train_dataloader['batch_size'] = 4

# 每轮都保存权重，并且只保留最新的权重
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))
# 将最大 epoch 数设置为 10，并每 1 个 epoch验证模型
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
#根据 10 个 epoch调整学习率调度
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1)
]
```

### 修改模型配置

此外，由于精简版 Kinetics 数据集规模较小，建议加载原始 Kinetics 数据集上的预训练模型。此外，模型需要根据实际类别数进行修改。请直接将以下代码添加到 `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` 中。

```python
model = dict(
    cls_head=dict(num_classes=2))
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'
```

在这里，我们直接通过继承 ({external+mmengine:doc} `MMEngine: Config <advanced_tutorials/ Config>`) 机制重写了基本配置中的相应参数。原始字段分布在 `configs/_base_/models/tsn_r50.py`、`configs/_base_/schedules/sgd_100e.py` 和 `configs/_base_/default_runtime.py`中。

```{note}
关于配置的更详细的描述，请参考[这里](../user_guides/config.md)。
```

## 浏览数据集

在开始训练之前，我们还可以将训练时数据转换处理的帧可视化。这很简单：传递我们需要可视化的配置文件到 [browse_dataset.py](/tools/analysis_tools/browse_dataset.py)脚本中。

```Bash
python tools/visualizations/browse_dataset.py \
    configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    browse_out --mode pipeline
```

转换后的视频将被保存到 `browse_out` 文件夹中。

<center class="half">
    <img src="https://user-images.githubusercontent.com/33249023/227452030-81895695-8a9b-45be-922a-3d9d86baf65d.gif" height="250"/>
</center>

```{note}
有关该脚本的参数和使用方法的详细信息，请参考[这里](../user_guides/useful_tools.md)。
```

```{tip}
除了满足我们的好奇心，可视化还可以帮助我们在训练前检查可能影响模型性能的部分，例如配置、数据集和数据转换中的问题。
```

我们可以通过以下脚本进一步可视化学习率调度，以确保配置符合预期:

```Bash
python tools/visualizations/vis_scheduler.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py
```

训练学习率时间表将显示在弹出窗口中。

<center class="half">
    <img src="https://user-images.githubusercontent.com/33249023/227502329-6fd44259-e23b-46e0-8e19-29f9b664f4e2.png" height="250"/>
</center>

```{note}
学习率根据实际批数据大小自动缩放。
```

## 训练

运行如下命令启动训练:

```Bash
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py
```

根据系统环境，MMAction2 将自动使用最佳设备进行训练。如果有GPU，则默认启动单个GPU训练。当你开始看到 loss 的输出时，就说明你已经成功启动了训练。

```Bash
03/24 16:36:15 - mmengine - INFO - Exp name: tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20230324_163608
03/24 16:36:15 - mmengine - INFO - Epoch(train)  [1][8/8]  lr: 1.5625e-04  eta: 0:00:15  time: 0.2151  data_time: 0.0845  memory: 1314  grad_norm: 8.5647  loss: 0.7267  top1_acc: 0.0000  top5_acc: 1.0000  loss_cls: 0.7267
03/24 16:36:16 - mmengine - INFO - Exp name: tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20230324_163608
03/24 16:36:16 - mmengine - INFO - Epoch(train)  [2][8/8]  lr: 1.5625e-04  eta: 0:00:12  time: 0.1979  data_time: 0.0717  memory: 1314  grad_norm: 8.4709  loss: 0.7130  top1_acc: 0.0000  top5_acc: 1.0000  loss_cls: 0.7130
03/24 16:36:18 - mmengine - INFO - Exp name: tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20230324_163608
03/24 16:36:18 - mmengine - INFO - Epoch(train)  [3][8/8]  lr: 1.5625e-04  eta: 0:00:10  time: 0.1691  data_time: 0.0478  memory: 1314  grad_norm: 8.2910  loss: 0.6900  top1_acc: 0.5000  top5_acc: 1.0000  loss_cls: 0.6900
03/24 16:36:18 - mmengine - INFO - Saving checkpoint at 3 epochs
03/24 16:36:19 - mmengine - INFO - Epoch(val) [3][1/1]  acc/top1: 0.9000  acc/top5: 1.0000  acc/mean1: 0.9000data_time: 1.2716  time: 1.3658
03/24 16:36:20 - mmengine - INFO - The best checkpoint with 0.9000 acc/top1 at 3 epoch is saved to best_acc/top1_epoch_3.pth.
```

在没有额外配置的情况下，模型权重将被保存到 `work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/`，而日志将被存储到 `work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/`。接下来，我们只需要耐心等待训练完成。

```{note}
训练的高级用法，如 CPU 训练、多卡训练及集群训练，请参考[training and Testing](../user_guides/train_test.md)
```

## 测试

经过 10 个 epoch 后，我们观察到 TSN 在第 6 个 epoch 表现最好，`acc/top1` 达到1.0000:

```Bash
03/24 16:36:25 - mmengine - INFO - Epoch(val) [6][1/1]  acc/top1: 1.0000  acc/top5: 1.0000  acc/mean1: 1.0000data_time: 1.0210  time: 1.1091
```

```{note}
由于在原始 Kinetics400 上进行了预训练，结果非常高，您可能会看到不同的结果
```

然而，该值仅反映了 TSN 在精简版 Kinetics 数据集上的验证性能，而测试结果通常更高，因为在测试数据流水线中增加了更多的数据增强。

开始测试：

```Bash
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc/top1_epoch_6.pth
```

并得到如下输出:

```Bash
03/24 17:00:59 - mmengine - INFO - Epoch(test) [10/10]  acc/top1: 1.0000  acc/top5: 1.0000  acc/mean1: 0.9000data_time: 0.0420  time: 1.0795
```

该模型在该数据集上实现了 1.000 的 top1 准确率。

```{note}
测试的高级用法，如CPU测试、多gpu测试、集群测试，请参考[Training and testing](../user_guides/train_test.md)
```
