# R2plus1D

## Model Zoo

### Kinetics-400

|config | backbone | pretrain| top1 acc| top5 acc | gpu_mem(M) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py) | ResNet34|None |68.68|88.36|5019|[ckpt]()| [log]()|
|[r2plus1d_r34_32x2x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb.py) | ResNet34|None |74.60|91.59|12975| [ckpt]() | [log]()|
|[r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py) | ResNet34|None |x|x|x| [ckpt]() | [log]()|

For more details on data preparation, you can refer to [preparing_kinetics400](/tools/data/kinetics400/preparing_kinetics400.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work_dir work_dirs/r2plus1d_r34_3d_8x8x1_180e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average_clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md).
