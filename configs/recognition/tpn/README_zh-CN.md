# TPN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|短边 320|8x2| ResNet50 | None | 73.58 | 91.35 | x | x | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb-c568e7ad.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.json) |
|[tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|短边 320|8| ResNet50 | ImageNet | 76.59 | 92.72 | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb-44362b55.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.json) |

### Something-Something V1

|配置文件 | GPU 数量 | 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_tsm_r50_1x1x8_150e_sthv1_rgb](/configs/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py)|height 100|8x6| ResNet50 | TSM | 51.50 | 79.15 | 8828 |[ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb_20211202-c28ed83f.pth) |[log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.log)|[json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.json)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。
4. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TPN 模型在 Kinetics-400 数据集上的训练。

```shell
python tools/train.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb [--validate --seed 0 --deterministic]
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 TPN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
