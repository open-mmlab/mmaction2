# PoseC3D

[重新审视基于骨架的动作识别](https://arxiv.org/abs/2104.13586)

<!-- [ALGORITHM] -->

## 摘要

<!-- [ABSTRACT] -->

人体骨架作为人类行为的一种紧凑表征，近年来受到越来越多的关注。许多基于骨架的动作识别方法采用图卷积网络（GCN）来提取人体骨架上的特征。尽管之前的研究取得了积极的成果，但基于 GCN 的方法在鲁棒性、互操作性和可扩展性方面仍受到限制。在这项工作中，我们提出了基于骨架的动作识别新方法 PoseC3D ，它依赖于 3D 热图堆叠而不是图形序列作为人体骨架的基本表示。与基于 GCN 的方法相比，PoseC3D 在学习时空特征方面更有效，对姿态估计噪声的鲁棒性更高，并且在跨数据集设置中具有更好的泛化能力。此外，PoseC3D 可以处理多人场景，无需额外的计算成本，并且其特征可以在早期融合阶段与其他模态轻松集成，这为进一步提升性能提供了巨大的设计空间。在四个具有挑战性的数据集上，PoseC3D 无论是单独用于骨架还是与 RGB 模式结合使用，都始终保持着卓越的性能。

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995620-21b5536c-8cda-48cd-9cb9-50b70cab7a89.png" width="800"/>
</div>

<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> 姿态估计结果 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529341-6fc95080-a90f-11eb-8f0d-57fdb35d1ba4.gif" width="455"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531676-04cd4900-a912-11eb-8db4-a93343bedd01.gif" width="455"/>
</div></td>
    <td>
<div align="center">
  <b> 关键点热图三维体可视化 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529336-6dff8d00-a90f-11eb-807e-4d9168997655.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531658-00a12b80-a912-11eb-957b-561c280a86da.gif" width="256"/>
</div></td>
    <td>
<div align="center">
  <b> 肢体热图三维体可视化 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529322-6a6c0600-a90f-11eb-81df-6fbb36230bd0.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531649-fed76800-a911-11eb-8ca9-0b4e58f43ad9.gif" width="256"/>
</div></td>
  </tr>
</thead>
</table>

## 结果和模型库

### FineGYM

| 帧提取策略 | 热图类型 | GPU数量 |   主干网络   | Mean Top-1 | 测试协议 | FLOPs | 参数量 |                   配置文件                    |                     ckpt                      |                     log                      |
| :--------: | :------: | :-----: | :----------: | :--------: | :------: | :---: | :----: | :-------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
| uniform 48 | keypoint |    8    | SlowOnly-R50 |    93.5    | 10 clips | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint_20220815-da338c58.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint.log) |
| uniform 48 |   limb   |    8    | SlowOnly-R50 |    93.6    | 10 clips | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb.log) |

### NTU60_XSub

| 帧提取策略 | 热图类型 | GPU数量 |   主干网络   | Mean Top-1 | 测试协议 | FLOPs | 参数量 |                   配置文件                    |                     ckpt                      |                     log                      |
| :--------: | :------: | :-----: | :----------: | :--------: | :------: | :---: | :----: | :-------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
| uniform 48 | keypoint |    8    | SlowOnly-R50 |    93.6    | 10 clips | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.log) |
| uniform 48 |   limb   |    8    | SlowOnly-R50 |    93.5    | 10 clips | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb_20220815-af2f119a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.log) |
|            |  Fusion  |         |              |    94.0    |          |       |        |                                               |                                               |                                              |

### UCF101

| 帧提取策略 | 热图类型 | GPU数量 |   主干网络   | Mean Top-1 | 测试协议 | FLOPs | 参数量 |                   配置文件                    |                     ckpt                      |                     log                      |
| :--------: | :------: | :-----: | :----------: | :--------: | :------: | :---: | :----: | :-------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
| uniform 48 | keypoint |    8    | SlowOnly-R50 |    86.8    | 10 clips | 14.6G |  3.1M  | [config](/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint_20220815-9972260d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.log) |

### HMDB51

| 帧提取策略 | 热图类型 | GPU数量 |   主干网络   | Mean Top-1 | 测试协议 | FLOPs | 参数量 |                   配置文件                    |                     ckpt                      |                     log                      |
| :--------: | :------: | :-----: | :----------: | :--------: | :------: | :---: | :----: | :-------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
| uniform 48 | keypoint |    8    | SlowOnly-R50 |    69.6    | 10 clips | 14.6G |  3.0M  | [config](/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.log) |

### Kinetics400

| 帧提取策略 | 热图类型 | GPU数量 |   主干网络   | Mean Top-1 | 测试协议 | FLOPs | 参数量 |                   配置文件                    |                     ckpt                      |                     log                      |
| :--------: | :------: | :-----: | :----------: | :--------: | :------: | :---: | :----: | :-------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
| uniform 48 | keypoint |    8    | SlowOnly-R50 |    47.4    | 10 clips | 19.1G |  3.2M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint/slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint/slowonly_r50_8xb32-u48-240e_k400-keypoint.log) |

用户可以参照 [准备骨架数据集](/tools/data/skeleton/README_zh-CN.md) 来获取以上配置文件使用的骨架标注。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: 以确定性的训练，进行 PoseC3D 模型在 FineGYM 数据集上的训练。

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    --seed=0 --deterministic
```

有关自定义数据集上的训练，可以参考 [Custom Dataset Training](/configs/skeleton/posec3d/custom_dataset_training.md)。

更多训练细节，可参考 [训练与测试](/docs/zh_cn/user_guides/train_test.md#%E8%AE%AD%E7%BB%83) 中的 **训练** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: 在 FineGYM 数据集上测试 PoseC3D 模型。

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth
```

更多训练细节，可参考 [训练与测试](/docs/zh_cn/user_guides/train_test.md#%E6%B5%8B%E8%AF%95) 中的 **测试** 部分。

## 引用

```BibTeX
@misc{duan2021revisiting,
      title={Revisiting Skeleton-based Action Recognition},
      author={Haodong Duan and Yue Zhao and Kai Chen and Dian Shao and Dahua Lin and Bo Dai},
      year={2021},
      eprint={2104.13586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
