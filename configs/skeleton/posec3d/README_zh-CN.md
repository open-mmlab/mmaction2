# PoseC3D

## 简介

<!-- [ALGORITHM] -->

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

## 模型库

### FineGYM

|配置文件 | 热图类型 | GPU 数量 | 主干网络 | Mean Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
|[slowonly_r50_u48_240e_gym_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py) | 关键点 |8 x 2| SlowOnly-R50 |93.7 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint-b07a98a0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint.json) |
|[slowonly_r50_u48_240e_gym_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb.py) | 肢体 |8 x 2| SlowOnly-R50 |94.0 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb-c0d7b482.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb.json) |
| 融合预测结果 | | |  |94.3 |  | | |

### NTU60_XSub

|配置文件 | 热图类型 | GPU 数量 | 主干网络 | Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
| [slowonly_r50_u48_240e_ntu60_xsub_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py) |    关键点    | 8 x 2 | SlowOnly-R50 | 93.7  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint.json) |
| [slowonly_r50_u48_240e_ntu60_xsub_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb.py) |      肢体      | 8 x 2 | SlowOnly-R50 | 93.4  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb-1d69006a.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb.json) |
| 融合预测结果                                                       |                |       |              | 94.1  |                                                              |                                                              |                                                              |

### NTU120_XSub

|配置文件 | 热图类型 | GPU 数量 | 主干网络 | Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
| [slowonly_r50_u48_240e_ntu120_xsub_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py) |    关键点    | 8 x 2 | SlowOnly-R50 | 86.3  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint.json) |
| [slowonly_r50_u48_240e_ntu120_xsub_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb.py) |      肢体      | 8 x 2 | SlowOnly-R50 | 85.7  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb-803c2317.pth?) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb.json) |
| 融合预测结果                                                       |                |       |              | 86.9  |                                                              |                                                              |                                                              |

### UCF101

|配置文件 | 热图类型 | GPU 数量 | 主干网络 | Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
| [slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint](/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py) |    关键点    | 8 | SlowOnly-R50 | 87.0  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint-cae8aa4a.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.json) |

### HMDB51

|配置文件 | 热图类型 | GPU 数量 | 主干网络 | Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
| [slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint](/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py) |    关键点    | 8 | SlowOnly-R50 | 69.3  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
  依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
  如，lr=0.2 对应 8 GPUs x 16 video/gpu，以及 lr=0.4 对应 16 GPUs x 16 video/gpu。
2. 用户可以参照 [准备骨骼数据集](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README_zh-CN.md) 来获取以上配置文件使用的骨骼标注。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: 以确定性的训练，加以定期的验证过程进行 PoseC3D 模型在 FineGYM 数据集上的训练。

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py \
    --work-dir work_dirs/slowonly_r50_u48_240e_gym_keypoint \
    --validate --seed 0 --deterministic
```

有关自定义数据集上的训练，可以参考 [Custom Dataset Training](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md)。

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: 在 FineGYM 数据集上测试 PoseC3D 模型，并将结果导出为一个 pickle 文件。

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
