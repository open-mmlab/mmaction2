# TimeSformer

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?},
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[timesformer_divST_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py) | 短边 320 | 8 | TimeSformer | ImageNet-21K | 77.92 | 93.29 | x | 17874 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb.json)|
|[timesformer_jointST_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb.py) | 短边 320 | 8 | TimeSformer | ImageNet-21K | 77.01 | 93.08 | x | 25658 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb-0d6e3984.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb.json)|
|[timesformer_sapceOnly_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_sapceOnly_8x32x1_15e_kinetics400_rgb.py) | 短边 320 | 8 | TimeSformer | ImageNet-21K | 76.93 | 92.90 | x | 12750 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb-0cf829cd.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb.json)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数 （32G V100）。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.005 对应 8 GPUs x 8 video/gpu，以及 lr=0.004375 对应 8 GPUs x 7 video/gpu。
2. MMAction2 保持与 [原代码](https://github.com/facebookresearch/TimeSformer) 的测试设置一致（three crop x 1 clip）。
3. TimeSformer 使用的预训练模型 `vit_base_patch16_224.pth` 转换自 [vision_transformer](https://github.com/google-research/vision_transformer)。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TimeSformer 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --work-dir work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 数据集上测试 TimeSformer 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
