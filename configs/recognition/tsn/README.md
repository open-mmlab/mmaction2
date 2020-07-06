# TSN

## Model Zoo

### UCF-101

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsn_r50_1x1x3_100e_ucf101_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb.py) | ResNet50 |80.12|96.09| x | x | [ckpt]() | [log]()|

### Kinetics-400

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) | ResNet50 |70.60|89.26| x | x | [ckpt]() | [log]()|
|[tsn_r50_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x5_50e_kinetics400_rgb.py) | ResNet50 |68.64|88.19| x | x | [ckpt]() | [log]()|
|[tsn_r50_dense_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb.py) | ResNet50 |68.59|88.31| x | x | [ckpt]() | [log]()|
|[tsn_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_100e_kinetics400_rgb.py) | ResNet50 |69.41|88.37| x | x | [ckpt]() | [log]()|
|[tsn_r50_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb.py) | ResNet50 |70.91|89.51| 8344 | x | [ckpt]() | [log]() |
|[tsn_r50_320p_1x1x3_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow.py) | ResNet50 |55.70|79.85| 8471 | x | [ckpt]() | [log]() |
|tsn_r50_320p_1x1x3_kinetics400_twostream [1: 1]* | ResNet50 |72.76|90.52| x | x | x | x |
|[tsn_r50_320p_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py) | ResNet50 |72.41|90.55| 8344       | x | [ckpt]() | [log]() |
|[tsn_r50_320p_1x1x8_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow.py) | ResNet50 |57.76|80.99| 8473 | x | [ckpt]() | [log]() |
|tsn_r50_320p_1x1x8_kinetics400_twostream [1: 1]* | ResNet50 |74.64|91.77| x | x | x | x |
|[tsn_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb.py) | ResNet50 |70.77|89.3| x | x | [ckpt]() | [log]()|
|[tsn_r50_video_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py) | ResNet50 | x | x | x | x | [ckpt]() | [log]()|

*We combine rgb and flow score with coefficients 1: 1 to get the two-stream prediction (without applying softmax).

### Something-Something V1

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py) | ResNet50 |18.55|44.80| x | x | [ckpt]() | [log]()|
|[tsn_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb.py) | ResNet50 |15.77|39.85| x | x | [ckpt]() | [log]()|

### Something-Something V2

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb.py) | ResNet50 |32.41|64.05| x | x | [ckpt]() | [log]()|
|[tsn_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py) | ResNet50 |22.48|49.08| x | x | [ckpt]() | [log]()|

### Moments in Time

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsn_r50_1x1x6_100e_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb.py) | ResNet50 |26.84|51.6| x | x | [ckpt]() | [log]()|

### Multi-Moments in Time

|config | pretrain | mAP| gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|
|[tsn_r101_1x1x5_50e_mmit_rgb](/configs/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb.py) | ResNet101 |61.09| x | x | [ckpt]() | [log]()|

For more details on data preparation, you can refer to [preparing_ucf101](/tools/data/ucf101/preparing_ucf101.md),
[preparing_kinetics400](/tools/data/kinetics400/preparing_kinetics400.md), [preparing_sthv1](/tools/data/sthv1/preparing_sthv1.md),
[preparing_sthv2](/tools/data/sthv2/preparing_sthv2.md), [preparing_mit](/tools/data/mit/preparing_mit.md),
[preparing_mmit](/tools/data/mmit/preparing_mmit.md).

## Train

You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSN model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    --work_dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md).

## Test

You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md).
