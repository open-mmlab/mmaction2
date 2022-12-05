# R2plus1D

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{tran2018closer,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={6450--6459},
  year={2018}
}
```

## 模型库

### Kinetics-400

| 配置文件                                                                                                                        |  分辨率  | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M) |                                                                                          ckpt                                                                                          |                                                                    log                                                                     |                                                                        json                                                                        |
| :------------------------------------------------------------------------------------------------------------------------------ | :------: | :------: | :------: | :----: | :---------: | :---------: | :----------------: | :--------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------: |
| [r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py)             | 短边 256 |   8x4    | ResNet34 |  None  |    67.30    |    87.65    |         x          |       5019       |  [ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb_20200729-aa94765e.pth)  |    [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/20200728_021421.log)    |     [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/20200728_021421.log.json)     |
| [r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py) | 短边 256 |    8     | ResNet34 |  None  |    67.3     |    87.8     |         x          |       5019       | [ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb_20200826-ab35a529.pth) | [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/20200724_201360.log.json) |       [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/20200724_201360.log)       |
| [r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py)             | 短边 320 |   8x2    | ResNet34 |  None  |    68.68    |    88.36    | 1.6 (80x3 frames)  |       5019       |       [ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth)       |          [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r21d_8x8.log)          | [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8_69.58_88.36.log.json) |
| [r2plus1d_r34_32x2x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb.py)           | 短边 320 |   8x2    | ResNet34 |  None  |    74.60    |    91.59    | 0.5 (320x3 frames) |      12975       |      [ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_kinetics400_rgb_20200618-63462eb3.pth)      |         [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r21d_32x2.log)         | [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2_74.6_91.6.log.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs/zh_cn/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 R(2+1)D 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work-dir work_dirs/r2plus1d_r34_3d_8x8x1_180e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 数据集上测试 R(2+1)D 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
