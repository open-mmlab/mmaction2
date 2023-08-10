# VideoSwin

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{liu2022video,
  title={Video swin transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3202--3211},
  year={2022}
}
```

## 模型库

### Kinetics-400

| 帧采样策略 | 分辨率  | GPU数量 | 主干网络 |    预训练    | top1 准确率 | top5 准确率 |   参考代码的 top1 准确率   |   参考代码的 top5准确率    |     测试协议     | 浮点运算数 | 参数量 |     配置文件      |       ckpt        |       log        |
| :--------: | :-----: | :-----: | :------: | :----------: | :---------: | :---------: | :------------------------: | :------------------------: | :--------------: | :--------: | :----: | :---------------: | :---------------: | :--------------: |
|   32x2x1   | 224x224 |    8    |  Swin-T  | ImageNet-1k  |    78.90    |    93.77    | 78.84 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 93.76 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop |    88G     | 28.2M  | [config](/configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|   32x2x1   | 224x224 |    8    |  Swin-S  | ImageNet-1k  |    80.54    |    94.46    | 80.58 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.45 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop |    166G    | 49.8M  | [config](/configs/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|   32x2x1   | 224x224 |    8    |  Swin-B  | ImageNet-1k  |    80.57    |    94.49    | 80.55 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.66 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop |    282G    | 88.0M  | [config](/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|   32x2x1   | 224x224 |    8    |  Swin-L  | ImageNet-22k |    83.46    |    95.91    |           83.1\*           |           95.9\*           | 4 clips x 3 crop |    604G    |  197M  | [config](/configs/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-78ad8b11.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |

### Kinetics-700

| 帧采样策略 | 分辨率  | GPU数量 | 主干网络 |    预训练    | top1 准确率 | top5 准确率 |     测试协议     | 浮点运算数 | 参数量 |              配置文件              |                ckpt                |                log                 |
| :--------: | :-----: | :-----: | :------: | :----------: | :---------: | :---------: | :--------------: | :--------: | :----: | :--------------------------------: | :--------------------------------: | :--------------------------------: |
|   32x2x1   | 224x224 |   16    |  Swin-L  | ImageNet-22k |    75.92    |    92.72    | 4 clips x 3 crop |    604G    |  197M  | [config](/configs/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.log) |

### Kinetics-710

| 帧采样策略 | 分辨率  | GPU数量 | 主干网络 |   预训练    | top1 准确率 | top5 准确率 |     测试协议     | 浮点运算数 | 参数量 |              配置文件              |                ckpt                 |                log                 |
| :--------: | :-----: | :-----: | :------: | :---------: | :---------: | :---------: | :--------------: | :--------: | :----: | :--------------------------------: | :---------------------------------: | :--------------------------------: |
|   32x2x1   | 224x224 |   32    |  Swin-S  | ImageNet-1k |    76.90    |    92.96    | 4 clips x 3 crop |    604G    |  197M  | [config](/configs/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb_20230612-8e082ff1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb.log) |

1. 这里的 **GPU数量** 指的是得到模型权重文件对应的 GPU 个数。当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要在运行 `tools/train.py` 时设置 `--auto-scale-lr` ，该参数将根据批大小等比例地调节学习率。
2. 参考代码的准确率列中的结果是通过使用相同的模型配置在原来的代码库上训练得到的。 `*` 代表数据来源于论文。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频。 用户可以从[验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB)下载这些视频。 同时也提供了对应的[数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (每行格式为：视频 ID，视频帧数目，类别序号) 以及[映射标签](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) 。
4. 预训练模型可以从 [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models)下载。

关于数据处理的更多细节，用户可以参照 [Kinetics](/tools/data/kinetics/README_zh-CN.md)。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 VideoSwin 模型在 Kinetics-400 数据集上的训练。

```shell
python tools/train.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

更多训练细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **训练** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 VideoSwin 模型，并将结果导出为一个 pkl 文件。

```shell
python tools/test.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

更多测试细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **测试** 部分。
