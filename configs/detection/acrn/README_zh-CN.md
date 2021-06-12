# ACRN

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{sun2018actor,
  title={Actor-centric relation network},
  author={Sun, Chen and Shrivastava, Abhinav and Vondrick, Carl and Murphy, Kevin and Sukthankar, Rahul and Schmid, Cordelia},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={318--334},
  year={2018}
}
```

## 模型库

### AVA2.1

|                            配置文件                             | 模态 |  预训练  | 主干网络 | 输入 | GPU 数量 | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb](/configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 27.1 | [log](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb-49b07bf2.pth) |

### AVA2.2

|                            配置文件                             | 模态 |  预训练  | 主干网络 | 输入 | GPU 数量 | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 27.8 | [log](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-2be32625.pth) |

- 注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。

对于数据集准备的细节，用户可参考 [数据准备](/docs_zh_CN/data_preparation.md)。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：在 AVA 数据集上训练 ACRN 辅以 SlowFast 主干网络，并定期验证。

```shell
python tools/train.py configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py --validate
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 AVA 上测试 ACRN 辅以 SlowFast 主干网络，并将结果存为 csv 文件。

```shell
python tools/test.py configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py checkpoints/SOME_CHECKPOINT.pth --eval mAP --out results.csv
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
