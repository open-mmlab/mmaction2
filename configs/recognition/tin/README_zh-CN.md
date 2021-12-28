# TIN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```

## 模型库

### Something-Something V1

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv1_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py)|高 100|8x4| ResNet50 | ImageNet | 44.25 | 73.94 | 44.04 | 72.72 | 6181 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/tin_r50_1x1x8_40e_sthv1_rgb_20200729-4a33db86.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log.json) |

### Something-Something V2

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv2_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py)|高 240|8x4| ResNet50 | ImageNet | 56.70 | 83.62 | 56.48 | 83.45 | 6185 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/tin_r50_1x1x8_40e_sthv2_rgb_20200912-b27a7337.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log.json) |

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络| 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb.py)|短边 256|8x4| ResNet50 | TSM-Kinetics400 | 70.89 | 89.89 | 6187 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb_20200810-4a146a70.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log.json) |

这里，MMAction2 使用 `finetune` 一词表示 TIN 模型使用 Kinetics400 上的 [TSM 模型](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) 进行微调。

注：

1. 参考代码的结果是通过 [原始 repo](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) 解决 [AverageMeter 相关问题](https://github.com/deepcs233/TIN/issues/4) 后训练得到的，该问题会导致错误的精度计算。
2. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
3. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
4. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。
5. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400, Something-Something V1 and Something-Something V2 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TIN 模型在 Something-Something V1 数据集上的训练。

```shell
python tools/train.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Something-Something V1 数据集上测试 TIN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
