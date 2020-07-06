# TSM

## Model Zoo

### Kinetics-400

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsm_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) | ResNet50 |70.24|89.56| x | x | [ckpt]() | [log]()|
|[tsm_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py) | ResNet50 |71.84|90.18| x | x | [ckpt]() | [log]()|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) | ResNet50 |71.69|90.4| x | x | [ckpt]() | [log]()|
|[tsm_r50_video_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb.py) | ResNet50 | x | x | x | x | [ckpt]() | [log]()|

### Something-Something V1

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsm_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py) | ResNet50 |44.62|75.51| x | x | [ckpt]() | [log]()|
|[tsm_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb.py) | ResNet50 |43.81|74.73| x | x | [ckpt]() | [log]()|
|[tsm_r101_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb.py) | ResNet50 |46.41|74.07| x | x | [ckpt]() | [log]()|

### Something-Something V2

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tsm_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py) | ResNet50 |59.91|84.61| x | x | [ckpt]() | [log]()|
|[tsm_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb.py) | ResNet50 |56.10|84.43| x | x | [ckpt]() | [log]()|
|[tsm_r101_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb.py) | ResNet50 | x | x | x | x | [ckpt]() | [log]()|

For more details on data preparation, you can refer to [preparing_kinetics400](/tools/data/kinetics400/preparing_kinetics400.md),
[preparing_sthv1](/tools/data/sthv1/preparing_sthv1.md), [preparing_sthv2](/tools/data/sthv2/preparing_sthv2.md)

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb.py \
    --work_dir work_dirs/tsm_r50_1x1x8_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md).
