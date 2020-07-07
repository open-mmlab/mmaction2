# R2plus1D

## Model Zoo

### Kinetics-400

|config | gpus | backbone | pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py) |8x2| ResNet34|None |68.68|88.36|1.6 (80x3 frames)|5019|[ckpt](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth)| [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r21d_8x8.log)| [json](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8_69.58_88.36.log.json)|
|[r2plus1d_r34_32x2x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb.py) |8x2| ResNet34|None |74.60|91.59|0.5 (320x3 frames)|12975| [ckpt](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_kinetics400_rgb_20200618-63462eb3.pth) | [log](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r21d_32x2.log)| [json](https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/mmaction-v1/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2_74.6_91.6.log.json)|
|[r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py) |x| ResNet34|None |x|x|x|x| [ckpt]() | [log]()| [json]()|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

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

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

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

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
