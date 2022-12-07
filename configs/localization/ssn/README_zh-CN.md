# SSN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@InProceedings{Zhao_2017_ICCV,
author = {Zhao, Yue and Xiong, Yuanjun and Wang, Limin and Wu, Zhirong and Tang, Xiaoou and Lin, Dahua},
title = {Temporal Action Detection With Structured Segment Networks},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

## 模型库

|                                         配置文件                                          | GPU 数量 | 主干网络 |  预训练  | mAP@0.3 | mAP@0.4 | mAP@0.5 |                                             参考代码的 mAP@0.3                                              |                                             参考代码的 mAP@0.4                                              |                                             参考代码的 mAP@0.5                                              | GPU 显存占用 (M) |                                                                    ckpt                                                                    |                                                      log                                                      | json                                                                                                                |                                                                            参考代码的 ckpt                                                                            |                                                              参考代码的 json                                                               |
| :---------------------------------------------------------------------------------------: | :------: | :------: | :------: | :-----: | :-----: | :-----: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: | :--------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| [ssn_r50_450e_thumos14_rgb](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) |    8     | ResNet50 | ImageNet |  29.37  |  22.15  |  15.69  | [27.61](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started) | [21.28](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started) | [14.57](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started) |       6352       | [ckpt](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/ssn_r50_450e_thumos14_rgb_20201012-1920ab16.pth) | [log](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log) | [json](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log.json) | [ckpt](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/ssn_r50_450e_thumos14_rgb_ref_20201014-b6f48f68.pth) | [json](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/20201008_103258.log.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 由于 SSN 在训练和测试阶段使用不同的结构化时序金字塔池化方法（structured temporal pyramid pooling methods），请分别参考 [ssn_r50_450e_thumos14_rgb_train](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) 和 [ssn_r50_450e_thumos14_rgb_test](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_test.py)。
3. MMAction2 使用 TAG 的时序动作候选进行 SSN 模型的精度验证。关于数据准备的更多细节，用户可参考 [Data 数据集准备文档](/docs/zh_cn/data_preparation.md) 准备 thumos14 的 TAG 时序动作候选。
4. 参考代码的 SSN 模型是和 MMAction2 一样在 `ResNet50` 主干网络上验证的。注意，这里的 SSN 的初始设置与原代码库的 `BNInception` 骨干网络的设置相同。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：在 thumos14 数据集上训练 SSN 模型。

```shell
python tools/train.py configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 ActivityNet 特征上测试 BMN。

```shell
# 注：如果需要进行指标验证，需确测试数据的保标注文件包含真实标签
python tools/test.py configs/localization/ssn/ssn_r50_450e_thumos14_rgb_test.py checkpoints/SOME_CHECKPOINT.pth --eval mAP
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
