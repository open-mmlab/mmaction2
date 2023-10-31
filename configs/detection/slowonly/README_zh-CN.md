# SlowOnly

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={ICCV},
  pages={6202--6211},
  year={2019}
}
```

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={CVPR},
  pages={6047--6056},
  year={2018}
}
```

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```


## 模型库

### AVA2.1

| 帧采样策略 | GPU数量 |                        主干网络                         |    预训练    |  mAP  |                  配置文件                  |                  ckpt                   |                  log                   |
| :-------: | :-----: | :----------------------------------------------------: | :----------: | :---: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|  4x16x1   |    8    |                   SlowOnly ResNet50                    | Kinetics-400 | 20.72 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-953ef5fe.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|  4x16x1   |    8    |                   SlowOnly ResNet50                    | Kinetics-700 | 22.77 | [config](/configs/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-b3b6d44e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|  4x16x1   |    8    | SlowOnly ResNet50 (NonLocalEmbedGauss - 非局部嵌入高斯) | Kinetics-400 | 21.55 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb_20220906-5ae3f91b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.log) |
|   8x8x1   |    8    | SlowOnly ResNet50 (NonLocalEmbedGauss - 非局部嵌入高斯) | Kinetics-400 | 23.77 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb_20220906-9760eadb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.log) |
|   8x8x1   |    8    |                   SlowOnly ResNet101                   | Kinetics-400 | 24.83 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb_20220906-43f16877.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |

### AVA2.2 (Trained on AVA-Kinetics)

目前我们仅使用 AVA-Kinetics 的训练集，并在 AVA2.2 的验证数据集上进行评估，AVA-Kinetics 的验证集将在不久的将来得到支持。

| 帧采样策略 | GPU数量 |     主干网络       |    预训练    |  mAP  |                      配置文件                     |                      ckpt                      |                      log                      |
| :-------: | :-----: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|  4x16x1   |    8    | SlowOnly ResNet50 | Kinetics-400 | 24.53 | [config](/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-33e3ca7c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|  4x16x1   |    8    | SlowOnly ResNet50 | Kinetics-700 | 25.87 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-a07e8c15.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|   8x8x1   |    8    | SlowOnly ResNet50 | Kinetics-400 | 26.10 | [config](/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-8f8dff3b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|   8x8x1   |    8    | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |

### AVA2.2 (Trained on AVA-Kinetics with tricks)

我们基于在Kinetics700 数据集上预训练的 SlowOnly8x8 进行了消融研究，以展示训练技巧的改进效果。 基线是 **AVA2.2 (Trained on AVA-Kinetics)** 中的最后一行。

|              方法                    | 帧采样策略 | GPU数量 |     主干网络      |    预训练     |  mAP  |                 配置文件                 |                  ckpt                   |                  log                   |
| :----------------------------------: | :-------: | :----: | :---------------: | :----------: | :---: | :--------------------------------------: | :-------------------------------------: | :------------------------------------: |
|           baseline (基准线)          |   8x8x1    |   8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|          + context (上下文)          |   8x8x1    |   8   | SlowOnly ResNet50 | Kinetics-700 | 28.31 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5d514f8c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
| + temporal max pooling (时域最大池化) |   8x8x1    |   8   | SlowOnly ResNet50 | Kinetics-700 | 28.48 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5b5e71eb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|     + nonlinear head (非线性头部)     |   8x8x1    |   8   | SlowOnly ResNet50 | Kinetics-700 | 29.83 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-87624265.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|        + focal loss (焦点损失)        |   8x8x1    |   8   | SlowOnly ResNet50 | Kinetics-700 | 30.33 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb_20221205-37aa8395.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.log) |
|        + more frames (额外帧)         |  16x4x     |   8   | SlowOnly ResNet50 | Kinetics-700 | 31.29 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb_20221205-dd652f81.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.log) |

### MultiSports

| 帧采样策略 | GPU数量 |      主干网络      |    预训练    | f-mAP | v-mAP@0.2 | v-mAP@0.5 | v-mAP@0.1:0.9 | GPU内存(M)  |               配置文件             |               ckpt               |               log                |
| :-------: | :-----: | :---------------: | :----------: | :---: | :-------: | :-------: | :-----------: | :--------: | :--------------------------------: | :------------------------------: | :------------------------------: |
|  4x16x1   |    8    | SlowOnly ResNet50 | Kinetics-400 | 26.40 |   15.48   |   10.62   |     9.65      |    8509    | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-a1ca5e76.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.log) |

1. 这里的 **GPU数量** 指的是得到模型权重文件对应的 GPU 个数。当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要在运行 `tools/train.py` 时设置 `--auto-scale-lr` ，该参数将根据批大小等比例地调节学习率。
2. **+ context** 表示同时使用 RoI 特征和全局池化特征进行分类; **+ temporal max pooling** 表示在特征的时间维度上使用最大池化; **nonlinear head** 表示使用一个两层的多层感知机来代替线性分类器。
3. MultiSports 数据集使用帧mAP(f-mAP) 和视频mAP(v-map) 来评估性能。帧mAP评估每帧的检测结果，而视频mAP使用 3D IoU 在多个阈值下评估管级别结果。详情可参阅 [比赛页面](https://codalab.lisn.upsaclay.fr/competitions/3736#learn_the_details-evaluation) 。

有关数据准备的更多详情，请参阅

- [AVA](/tools/data/ava/README_zh-CN.md)
- [AVA-Kinetics](/tools/data/ava_kinetics/README_zh-CN.md)
- [MultiSports](/tools/data/multisports/README.md)

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

示例： 以一个确定性的训练方式，辅以定期的验证过程进行 SlowOnly 模型在 AVA2.1 数据集上的训练。

```shell
python tools/train.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

更多训练细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **训练** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

示例: 在 AVA2.1 数据集上测试 SlowOnly 模型，并将结果导出为一个 pkl 文件。

```shell
python tools/test.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

更多测试细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **测试** 部分。
