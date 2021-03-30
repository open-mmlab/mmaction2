# TRN

## 简介

[ALGORITHM]

```BibTeX
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

## 模型库

### Something-Something V1

|配置文件 | 分辨率 | GPU 数量 | 主干网络| 预训练 | top1 准确率 (efficient/accurate)| top5 准确率 (efficient/accurate)| GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv1_rgb](configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py) | 高 100 | 8 | ResNet50 | ImageNet | 31.62 / 33.88 |60.01 / 72.12| 11010 | [ckpt](waiting for url) | [log](waiting for url)| [json](waiting for url)|

### Something-Something V2

|配置文件 | 分辨率 | GPU 数量 | 主干网络| 预训练 | top1 准确率 (efficient/accurate)| top5 准确率 (efficient/accurate)| GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv2_rgb](configs/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py) | 高 100 | 8 | ResNet50 | ImageNet | 45.14 / 47.96 |73.21 / 75.97 | 11010 | [ckpt](waiting for url) | [log](waiting for url)| [json](waiting for url)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 对于 Something-Something 数据集，有两种测试方案：efficient（对应 center crop x 1 clip）和 accurate（对应 Three crop x 2 clip）。
   MMAction2 使用 efficient 方案作为配置文件中的默认选择，用户可以通过以下方式转变为 accurate 方案：

```python
...
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,   # 当使用 8 个 视频段时，设置 `num_clips = 8`
        twice_sample=True,    # 设置 `twice_sample=True` 用于 accurate 方案中的 Twice Sample
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224), 用于 efficient 方案
    dict(type='ThreeCrop', crop_size=256),  # 用于 accurate 方案
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
```

关于数据处理的更多细节，用户可以参照

- [准备 sthv1](/tools/data/sthv1/README_zh-CN.md)
- [准备 sthv2](/tools/data/sthv2/README_zh-CN.md)

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TRN 模型在 sthv1 数据集上的训练。

```shell
python tools/train.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 sthv1 数据集上测试 TRN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
