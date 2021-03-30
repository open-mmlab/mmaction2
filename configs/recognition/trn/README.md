# TRN

## Introduction

[ALGORITHM]

```BibTeX
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

## Model Zoo

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv1_rgb](configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py) | height 100 | 8 | ResNet50 | ImageNet | 31.62 / 33.88 |60.01 / 72.12| 11010 | [ckpt](waiting for url) | [log](waiting for url)| [json](waiting for url)|

### Something-Something V2

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv2_rgb](configs/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py) | height 100 | 8 | ResNet50 | ImageNet | 45.14 / 47.96 |73.21 / 75.97 | 11010 | [ckpt](waiting for url) | [log](waiting for url)| [json](waiting for url)|

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip).
   We use efficient setting as default provided in config files, and it can be changed to accurate setting by

```python
...
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,    # set `twice_sample=True` for twice sample in accurate setting
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224), it is used for efficient setting
    dict(type='ThreeCrop', crop_size=256),  # it is used for accurate setting
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
```

For more details on data preparation, you can refer to

- [preparing_sthv1](/tools/data/sthv1/README.md)
- [preparing_sthv2](/tools/data/sthv2/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TRN model on sthv1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TRN model on sthv1 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset#test-a-dataset).
