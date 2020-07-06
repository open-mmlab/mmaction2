# TIN

## Model Zoo

### Kinetics-400

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tin_r50_1x1x8_35e_kinetics400_rgb](/configs/recognition/tin/tin_r50_1x1x8_35e_kinetics400_rgb.py) | ResNet50 |69.44|89.19| x | x | [ckpt]() | [log]()|
|[tin_r50_finetune_1x1x8_35e_kinetics400_rgb](/configs/recognition/tin/tin_r50_finetune_1x1x8_35e_kinetics400_rgb.py) | TSM |71.00|89.98| x | x | [ckpt]() | [log]()|
|[tin_r50_video_2d_1x1x8_35e_kinetics400_rgb](/configs/recognition/tin/tin_r50_video_1x1x8_35e_kinetics400_rgb.py) | ResNet50 | x | x | x | x | [ckpt]() | [log]()|

### Something-Something V1

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tin_r50_1x1x8_35e_sthv1_rgb](/configs/recognition/tin/tin_r50_1x1x8_35e_sthv1_rgb.py) | ResNet50 |41.59|71.94| x | x | [ckpt]() | [log]()|

### Something-Something V2

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[tin_r50_1x1x8_35e_sthv2_rgb](/configs/recognition/tin/tin_r50_1x1x8_35e_sthv2_rgb.py) | ResNet50 |53.08|82.02| x | x | [ckpt]() | [log]()|

For more details on data preparation, you can refer to [preparing_kinetics400](/tools/data/kinetics400/preparing_kinetics400.md),
[preparing_sthv1](/tools/data/sthv1/preparing_sthv1.md), [preparing_sthv2](/tools/data/sthv2/preparing_sthv2.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TIN model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tin/tin_r50_1x1x8_35e_kinetics400_rgb.py \
    --work_dir work_dirs/tin_r50_1x1x8_35e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TIN model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tin/tin_r50_1x1x8_35e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md).
