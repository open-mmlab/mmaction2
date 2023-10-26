# STGCN++

[PYSKL: 实现骨架动作识别的良好实践](https://arxiv.org/abs/2205.09443)

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

我们介绍PYSKL：一个基于 PyTorch 的骨架动作识别开源工具箱。该工具箱支持多种骨架动作识别算法，包括基于 GCN 和 CNN 的方法。与现有仅包含一两种算法的开源骨架动作识别项目相比，PYSKL 在统一框架下实现了六种不同的算法，并结合了最新和原创的优秀实践，便于比较功效和效率。我们还提供了一个基于 GCN 的原始骨架动作识别模型 ST-GCN++，该模型无需任何复杂的注意力方案即可实现有竞争力的识别性能，可作为一个强有力的基准。同时，PYSKL 支持九种基于骨架的动作识别基准的训练和测试，并在其中八种基准上实现了最先进的识别性能。为了方便未来的骨架动作识别研究，我们还提供了大量的训练模型和详细的基准测试结果，以提供一些见解。PYSKL发布在这个https网址上，并得到了积极的维护。我们将在添加新功能或基准后更新本报告。当前版本对应 PYSKL v0.2。

## 结果和模型库

### NTU60_XSub_2D

| 帧提取策略 |   模式   | GPU数量 | 主干网络 | Top-1 准确率 | 测试协议 | FLOPs | 参数量 |                  配置文件                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.29   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  92.30   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  87.30   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  88.76   |     10 clips     | 1.95G | 1.39M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  92.61   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.77   |                  |       |        |                                           |                                         |                                        |

### NTU60_XSub_3D

| 帧提取策略 |   模式   | GPU数量 | 主干网络 | Top-1 准确率 | 测试协议 | FLOPs | 参数量 |                  配置文件                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.14   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221230-4e455ce3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  90.21   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221230-7f356072.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  86.67   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-650de5cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  87.45   |     10 clips     | 2.96G |  1.4M  | [config](/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-b00440d2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  91.39   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  91.87   |                  |       |        |                                           |                                         |                                        |

1. 这里的 **GPU** 数量 指的是得到模型权重文件对应的 GPU 个数。用户在 使用不同数量的 GPU 或每块 GPU 处理不同视频个数时，最好的方法是在调用 `tools/train.py` 时设置 `--auto-scale-lr` ，该参数将根据实际批次大小自动调整学习率和原始批次。
1. 对于双流融合，我们使用 **joint : bone = 1 : 1**。对于四流融合，我们使用**joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**。有关多流融合的更多详情信息，请参考[教程](/docs/zh_cn/useful_tools.md#%E5%A4%9A%E6%B5%81%E8%9E%8D%E5%90%88)。

## 训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: 以确定性的训练，进行 STGCN++ 模型在 NTU60-2D 数据集上的训练。

```shell
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

更多训练细节，可参考 [训练与测试](/docs/zh_cn/user_guides/train_test.md#%E8%AE%AD%E7%BB%83) 中的 **训练** 部分。

## 测试

用户可以使用以下指令进行模型训练。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: 以确定性的训练，加以定期的验证过程进行 STGCN++ 模型在 NTU60-2D 数据集上的训练。

```shell
python tools/test.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

更多训练细节，可参考 [训练与测试](/docs/zh_cn/user_guides/train_test.md#%E6%B5%8B%E8%AF%95) 中的 **测试** 部分。

## 引用

```BibTeX
@misc{duan2022PYSKL,
  url = {https://arxiv.org/abs/2205.09443},
  author = {Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  title = {PYSKL: Towards Good Practices for Skeleton Action Recognition},
  publisher = {arXiv},
  year = {2022}
}
```
