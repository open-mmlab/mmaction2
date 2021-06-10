# SlowFast

## 简介

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

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 |预训练| top1 准确率| top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py) |短边256|8x4| ResNet50|None |74.75|91.73|x|6203|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/20200731_151706.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/20200731_151706.log.json)|
|[slowfast_r50_video_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py) |短边256|8| ResNet50|None |74.34|91.58|x|6203|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb/slowfast_r50_video_4x16x1_256e_kinetics400_rgb_20200826-f85b90c5.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb/20200812_160237.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb/20200812_160237.log.json)|
|[slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py) |短边320|8x3| ResNet50|None |75.64|92.3|1.6 ((32+4)x10x3 frames)|6203|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200704-bcde7ed7.pth)| [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/20200704_232901.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/20200704_232901.log.json)|
|[slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py) |短边256|8x4| ResNet50 |None |75.61|92.34|x|9062|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb_20200810-863812c2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb/20200731_151537.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_8x8x1_256e_kinetics400_rgb/20200731_151537.log.json)|
|[slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py) |短边320|8x3| ResNet50 |None|76.94|92.8|1.3 ((32+8)x10x3 frames)|9062| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/20200716_192653.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/20200716_192653.log.json)|
|[slowfast_r101_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r101_r50_4x16x1_256e_kinetics400_rgb.py) |短边256|8x1| ResNet101 + ResNet50 |None|76.69|93.07||16628| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/20210118_133528.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/20210118_133528.log.json)|
|[slowfast_r101_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py) |短边256|8x4| ResNet101 |None|77.90|93.51||25994| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/20210218_121513.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/20210218_121513.log.json)|
|[slowfast_r152_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r152_r50_4x16x1_256e_kinetics400_rgb.py) |短边256|8x1| ResNet152 + ResNet50 |None|77.13|93.20||10077| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r152_4x16x1_256e_kinetics400_rgb/slowfast_r152_4x16x1_256e_kinetics400_rgb_20210122-bdeb6b87.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r152_4x16x1_256e_kinetics400_rgb/20210122_131321.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r152_4x16x1_256e_kinetics400_rgb/20210122_131321.log.json)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 SlowFast 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowfast_r50_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 SlowFast 数据集上测试 CSN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
