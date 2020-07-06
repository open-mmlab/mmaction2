# SlowFast

## Model Zoo

### Kinetics-400

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py) | None |75.3|92.2|6250|0.826|[ckpt]()| [log]()|
|[slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py) | None |76.36|92.56|9159|1.032| [ckpt]() | [log]()|
|[slowfast_r50_video_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py) | None |x|x|x|x| [ckpt]() | [log]()|

For more details on data preparation, you can refer to [preparing_kinetics400](/tools/data/kinetics400/preparing_kinetics400.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_3d_4x16x1_256e_kinetics400_rgb.py \
    --work_dir work_dirs/slowfast_r50_3d_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](../../../docs/getting_started.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_3d_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average_clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](../../../docs/getting_started.md).
