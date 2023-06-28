# Quick Run

This chapter will introduce you to the fundamental functionalities of MMAction2. We assume that you have [installed MMAction2 from source](installation.md#best-practices).

- [Quick Run](#quick-run)
  - [Inference](#inference)
  - [Prepare a Dataset](#prepare-a-dataset)
  - [Modify the Config](#modify-the-config)
    - [Modify Dataset](#modify-dataset)
    - [Modify Runtime Config](#modify-runtime-config)
    - [Modify Model Config](#modify-model-config)
  - [Browse the Dataset](#browse-the-dataset)
  - [Training](#training)
  - [Testing](#testing)

## Inference

Run the following command in the root directory of MMAction2:

```shell
python demo/demo_inferencer.py  demo/demo.mp4 \
    --rec tsn --print-result \
    --label-file tools/data/kinetics/label_map_k400.txt
```

You should be able to see a pop-up video and the inference result printed out in the console.

<div align="center">
    <img src="https://user-images.githubusercontent.com/33249023/227216933-29b84ac7-ca0e-408d-b4d2-5a2e5a7357bf.gif" height="250"/>
</div>
<br />

```bash
# Inference result
{'predictions': [{'rec_labels': [[6]], 'rec_scores': [[...]]}]}
```

```{note}
If you are running MMAction2 on a server without a GUI or via an SSH tunnel with X11 forwarding disabled, you may not see the pop-up window.
```

A detailed description of MMAction2's inference interface can be found [here](https://github.com/open-mmlab/mmaction2/tree/main/demo/README.md#inferencer).

In addition to using our well-provided pre-trained models, you can also train models on your own datasets. In the next section, we will take you through the basic functions of MMAction2 by training TSN on the tiny [Kinetics](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) dataset as an example.

## Prepare a Dataset

Since the variety of video dataset formats are not conducive to switching datasets, MMAction2 proposes a uniform [data format](../user_guides/2_data_prepare.md), and provides [dataset preparer](../user_guides/data_prepare/dataset_preparer.md) for commonly used video datasets. Usually, to use those datasets in MMAction2, you just need to follow the steps to get them ready for use.

```{note}
But here, efficiency means everything.
```

To get started, please download our pre-prepared [kinetics400_tiny.zip](https://download.openmmlab.com/mmaction/kinetics400_tiny.zip) and extract it to the `data/` directory in the root directory of MMAction2. This will provide you with the necessary videos and annotation file.

```Bash
wget https://download.openmmlab.com/mmaction/kinetics400_tiny.zip
mkdir -p data/
unzip kinetics400_tiny.zip -d data/
```

## Modify the Config

After preparing the dataset, the next step is to modify the config file to specify the location of the training set and training parameters.

In this example, we will train a TSN using resnet50 as its backbone. Since MMAction2 already has a config file for the full Kinetics400 dataset (`configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`), we just need to make some modifications on top of it.

### Modify Dataset

We first need to modify the path to the dataset. Open `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` and replace keys as followed:

```Python
data_root = 'data/kinetics400_tiny/train'
data_root_val = 'data/kinetics400_tiny/val'
ann_file_train = 'data/kinetics400_tiny/kinetics_tiny_train_video.txt'
ann_file_val = 'data/kinetics400_tiny/kinetics_tiny_val_video.txt'
```

### Modify Runtime Config

Additionally, due to the reduced size of the dataset, we recommend decreasing the training batch size to 4 and the number of training epochs to 10 accordingly. Furthermore, we suggest shortening the validation and weight storage intervals to 1 round each, and modifying the learning rate decay strategy. Modify the corresponding keys in  `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py` as following lines to take effect.

```python
# set training batch size to 4
train_dataloader['batch_size'] = 4

# Save checkpoints every epoch, and only keep the latest checkpoint
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))
# Set the maximum number of epochs to 10, and validate the model every 1 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
# adjust learning rate schedule according to 10 epochs
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

### Modify Model Config

Further, due to the small size of tiny Kinetics dataset, it is recommended to load a pre-trained model on the original Kinetics dataset. Additionally, the model needs to be modified according to the actual number of classes. Please directly add the following lines to `configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`.

```python
model = dict(
    cls_head=dict(num_classes=2))
load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth'
```

Here, we have rewritten the corresponding parameters in the base configuration directly through the inheritance ({external+mmengine:doc}`MMEngine: Config <advanced_tutorials/config>`) mechanism of the config. The original fields are distributed in `configs/_base_/models/tsn_r50.py`, `configs/_base_/schedules/sgd_100e.py` and `configs/_base_/default_runtime.py`.

```{note}
For a more detailed description of config, please refer to [here](../user_guides/1_config.md).
```

## Browse the Dataset

Before we start the training, we can also visualize the frames processed by training-time data transforms. It's quite simple: pass the config file we need to visualize into the [browse_dataset.py](https://github.com/open-mmlab/mmaction2/tree/main/tools/analysis_tools/browse_dataset.py) script.

```Bash
python tools/visualizations/browse_dataset.py \
    configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    browse_out --mode pipeline
```

The transformed videos will be saved to `browse_out` folder.

<center class="half">
    <img src="https://user-images.githubusercontent.com/33249023/227452030-81895695-8a9b-45be-922a-3d9d86baf65d.gif" height="250"/>
</center>

```{note}
For details on the parameters and usage of this script, please refer to [here](../user_guides/useful_tools.md).
```

```{tip}
In addition to satisfying our curiosity, visualization can also help us check the parts that may affect the model's performance before training, such as problems in configs, datasets and data transforms.
```

we can further visualize the learning rate schedule to make sure that the config is as expected by following script:

```Bash
python tools/visualizations/vis_scheduler.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py
```

The training learning rate schedule will be displayed in a pop-up window.

<center class="half">
    <img src="https://user-images.githubusercontent.com/33249023/227502329-6fd44259-e23b-46e0-8e19-29f9b664f4e2.png" height="250"/>
</center>

```{note}
The learning rate is auto scaled according to the actual batchsize.
```

## Training

Start the training by running the following command:

```Bash
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py
```

Depending on the system environment, MMAction2 will automatically use the best device for training. If a GPU is available, a single GPU training will be started by default. When you start to see the output of the losses, you have successfully started the training.

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

Without extra configurations, model weights will be saved to `work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/`, while the logs will be stored in `work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/TIMESTAMP/`. Next, we just need to wait with some patience for training to finish.

```{note}
For advanced usage of training, such as CPU training, multi-GPU training, and cluster training, please refer to [Training and Testing](../user_guides/train_test.md).
```

## Testing

After 10 epochs, we observe that TSN performs best in the 6th epoch, with `acc/top1` reaching 1.0000:

```Bash
03/24 16:36:25 - mmengine - INFO - Epoch(val) [6][1/1]  acc/top1: 1.0000  acc/top5: 1.0000  acc/mean1: 1.0000data_time: 1.0210  time: 1.1091
```

```{note}
The result is pretty high due to pre-trained on original Kinetics400, you may see a different result.
```

However, this value only reflects the validation performance of TSN on the mini Kinetics dataset, While test results are usually higher due to more augmentation in test pipeline.

Start testing:

```Bash
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc/top1_epoch_6.pth
```

And get the outputs like:

```Bash
03/24 17:00:59 - mmengine - INFO - Epoch(test) [10/10]  acc/top1: 1.0000  acc/top5: 1.0000  acc/mean1: 0.9000data_time: 0.0420  time: 1.0795
```

The model achieves an top1-accuracy of 1.0000 on this dataset.

```{note}
For advanced usage of testing, such as CPU testing, multi-GPU testing, and cluster testing, please refer to [Training and Testing](../user_guides/train_test.md).
```
