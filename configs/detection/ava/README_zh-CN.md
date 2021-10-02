# AVA

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/>
</div>

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
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## 模型库

### AVA2.1

|                            配置文件                         | 模态 |  预训练  | 主干网络  | 输入 | GPU 数量 |   分辨率   | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | 短边 256 | 20.1 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) |
| [slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet50  | 4x16  |  8   | 短边 256 | 21.8 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201217-0c6d2e98.pth) |
| [slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | 短边 256 | 21.75 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/20210316_122517.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/20210316_122517.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb_20210316-959829ec.pth) |
| [slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 8x8  |  8x2   | 短边 256 | 23.79 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/20210316_122517.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/20210316_122517.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb_20210316-5742e4dd.pth) |
| [slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet101 |  8x8  | 8x2  | 短边 256 | 24.6 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201217-1c9b4117.pth) |
| [slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet101 |  8x8  | 8x2  | 短边 256 | 25.9 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth) |
| [slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | 短边 256 | 24.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth) |
| [slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | 短边 256 | 25.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222-f4d209c9.pth) |
| [slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | 短边 256 | 25.5 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth) |

### AVA2.2

|                           配置文件                           | 模态 |    预训练    | 主干网络 | 输入 | GPU 数量 | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :--: | :----------: | :------: | :--: | :------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) | RGB  | Kinetics-400 | ResNet50 | 32x2 |    8     | 26.1 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-b987b516.pth) |
| [slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) | RGB  | Kinetics-400 | ResNet50 | 32x2 |    8     | 26.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-874e0845.pth) |
| [slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) | RGB  | Kinetics-400 | ResNet50 | 32x2 |    8     | 26.8 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. **Context** 表示同时使用 RoI 特征与全局特征进行分类，可带来约 1% mAP 的提升。

对于数据集准备的细节，用户可参考 [数据准备](/docs_zh_CN/data_preparation.md)。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：在 AVA 数据集上训练 SlowOnly，并定期验证。

```shell
python tools/train.py configs/detection/ava/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py --validate
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

### 训练 AVA 数据集中的自定义类别

用户可以训练 AVA 数据集中的自定义类别。AVA 中不同类别的样本量很不平衡：其中有超过 100000 样本的类别： `stand`/`listen to (a person)`/`talk to (e.g., self, a person, a group)`/`watch (a person)`，也有样本较少的类别（半数类别不足 500 样本）。大多数情况下，仅使用样本较少的类别进行训练将在这些类别上得到更好精度。

训练 AVA 数据集中的自定义类别包含 3 个步骤：

1. 从原先的类别中选择希望训练的类别，将其填写至配置文件的 `custom_classes` 域中。其中 `0` 不表示具体的动作类别，不应被选择。
2. 将 `num_classes` 设置为 `num_classes = len(custom_classes) + 1`。
   - 在新的类别到编号的对应中，编号 `0` 仍对应原类别 `0`，编号 `i` (i > 0) 对应原类别 `custom_classes[i-1]`。
   - 配置文件中 3 处涉及 `num_classes` 需要修改：`model -> roi_head -> bbox_head -> num_classes`， `data -> train -> num_classes`， `data -> val -> num_classes`.
   - 若 `num_classes <= 5`， 配置文件 `BBoxHeadAVA` 中的 `topk` 参数应被修改。`topk` 的默认值为 `(3, 5)`，`topk` 中的所有元素应小于 `num_classes`。
3. 确认所有自定义类别在 `label_file` 中。

以 `slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb` 为例，这一配置文件训练所有 AP 在 `(0.1, 0.3)` 间的类别（这里的 AP 为 AVA 80 类训出模型的表现），即 `[3, 6, 10, 27, 29, 38, 41, 48, 51, 53, 54, 59, 61, 64, 70, 72]`。下表列出了自定义类别训练的模型精度：

|训练类别|mAP （自定义类别）|配置文件|log|json|ckpt|
|:-:|:-:|:-:|:-:|:-:|:-:|
|全部 80 类|0.1948|[slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py)|[log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) |
|自定义类别|0.3311|[slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes-4ab80419.pth) |
|全部 80 类|0.1864|[slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth) |
|自定义类别|0.3785|[slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305-c6225546.pth) |

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 AVA 上测试 SlowOnly 模型，并将结果存为 csv 文件。

```shell
python tools/test.py configs/detection/ava/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py checkpoints/SOME_CHECKPOINT.pth --eval mAP --out results.csv
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
