# TANet

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@article{liu2020tam,
  title={TAM: Temporal Adaptive Module for Video Recognition},
  author={Liu, Zhaoyang and Wang, Limin and Wu, Wayne and Qian, Chen and Lu, Tong},
  journal={arXiv preprint arXiv:2005.06803},
  year={2020}
}
```

## 模型库

### Kinetics-400

| 配置文件                                                                                                               |  分辨率  | GPU 数量 | 主干网络 |  预训练  | top1 准确率 | top5 准确率 |                                            参考代码的 top1 准确率                                            |                                            参考代码的 top5 准确率                                            | 推理时间 (video/s) | GPU 显存占用 (M) |                                                                                     ckpt                                                                                      |                                                                                 log                                                                                 |                                                                                 json                                                                                  |
| :--------------------------------------------------------------------------------------------------------------------- | :------: | :------: | :------: | :------: | :---------: | :---------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :----------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [tanet_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py) | 短边 320 |    8     |  TANet   | ImageNet |    76.28    |    92.60    | [76.22](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | [92.53](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) |         x          |       7124       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219.log) | [json](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219.json) |

### Something-Something V1

| 配置文件                                                                                       | 分辨率 | GPU 数量 | 主干网络 |  预训练  | top1 准确率 (efficient/accurate) | top5 准确率 (efficient/accurate) | GPU 显存占用 (M) |                                                                         ckpt                                                                          |                                                                log                                                                 |                                                                 json                                                                 |
| :--------------------------------------------------------------------------------------------- | :----: | :------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| [tanet_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb.py)   | 高 100 |    8     |  TANet   | ImageNet |           47.34/49.58            |           75.72/77.31            |       7127       |  [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/tanet_r50_1x1x8_50e_sthv1_rgb_20210630-f4a48609.pth)  |         [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/20210606_205006.log)         |       [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/20210606_205006.log.json)       |
| [tanet_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb.py) | 高 100 |    8     |  TANet   | ImageNet |           49.05/50.91            |           77.90/79.13            |       7127       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb_20211202-370c2128.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb.log) | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。对应的模型权重文件可从 [这里](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing) 下载。
4. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs/zh_cn/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TANet 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 数据集上测试 TANet 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
