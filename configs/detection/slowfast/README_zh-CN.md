# SlowFast

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


## 模型库

### AVA2.1

| 帧采样策略 | GPU数量 |             主干网络              |    预训练    |  mAP  |                   配置文件                    |                   ckpt                    |                   log                    |
| :-------: | :-----: | :------------------------------: | :----------: | :---: | :-----------------------------------------: | :---------------------------------------: | :--------------------------------------: |
|  4x16x1   |    8    |        SlowFast ResNet50         | Kinetics-400 | 24.32 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-5180ea3c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|  4x16x1   |    8    | SlowFast ResNet50 (with context - 带有上下文) | Kinetics-400 | 25.34 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb_20220906-5bb4f6f2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.log) |
|   8x8x1   |    8    |        SlowFast ResNet50         | Kinetics-400 | 25.80 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.log) |

### AVA2.2

| 帧采样策略 | GPU数量 |                 主干网络                   |    预训练    |  mAP  |                  配置文件                  |                  ckpt                  |                  log                  |
| :-------: | :-----: | :---------------------------------------: | :----------: | :---: | :--------------------------------------: | :------------------------------------: | :-----------------------------------: |
|   8x8x1   |    8    |             SlowFast ResNet50             | Kinetics-400 | 25.90 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|   8x8x1   |    8    |     SlowFast ResNet50 (temporal-max - 时域最大池化)      | Kinetics-400 | 26.41 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-13a9078e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|   8x8x1   |    8    | SlowFast ResNet50 (temporal-max, focal loss - 时域最大池化，焦点损失) | Kinetics-400 | 26.65 | [config](/configs/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-dd59e26f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |

### MultiSports

| 帧采样策略 | GPU数量 |     主干网络       |    预训练    | f-mAP | v-mAP@0.2 | v-mAP@0.5 | v-mAP@0.1:0.9 | GPU内存(M)  |               配置文件              |               ckpt               |               log                |
| :-------: | :-----: | :---------------: | :----------: | :---: | :-------: | :-------: | :-----------: | :--------: | :--------------------------------: | :------------------------------: | :------------------------------: |
|  4x16x1   |    8    | SlowFast ResNet50 | Kinetics-400 | 36.88 |   22.83   |   16.9    |     14.74     |   18618    | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-af666368.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.log) |

1. 这里的 **GPU数量** 指的是得到模型权重文件对应的 GPU 个数。当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要在运行 `tools/train.py` 时设置 `--auto-scale-lr` ，该参数将根据批大小等比例地调节学习率。
2. **with context** 表示同时使用 RoI 特征和全局池化特征进行分类; **temporal-max** 表示在特征的时间维度上使用最大池化。
3. MultiSports 数据集使用帧mAP(f-mAP) 和视频mAP(v-map) 来评估性能。帧mAP评估每帧的检测结果，而视频mAP使用 3D IoU 在多个阈值下评估管级别结果。 详情可参阅 [比赛页面](https://codalab.lisn.upsaclay.fr/competitions/3736#learn_the_details-evaluation) 。

有关数据准备的更多详情，请参阅

- [AVA](/tools/data/ava/README_zh-CN.md)
- [MultiSports](/tools/data/multisports/README_zh-CN.md)

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

示例： 以一个确定性的训练方式，辅以定期的验证过程进行 SlowFast 模型在 AVA2.1 数据集上的训练。

```shell
python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

更多训练细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **训练** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

示例: 在 AVA2.1 数据集上测试 SlowFast 模型，并将结果导出为一个 pkl 文件。

```shell
python tools/test.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

更多测试细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **测试** 部分。
