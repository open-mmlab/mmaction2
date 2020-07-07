# SlowOnly

## Model Zoo

### Kinetics-400

|config | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)|8x2| ResNet50 | None |73.02|90.77|4.0 (40x3 frames)|3168|[ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth)| [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log)| [json](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json)|
|[slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) |8x3| ResNet50 | None |74.93|91.92|2.3 (80x3 frames)|5820| [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/so_8x8.log)| [json](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8_74.93_91.92.log.json)|
|[slowonly_r50_4x16x1_256e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow.py)|8x2| ResNet50  | ImageNet |61.79|83.62|x|8450| [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log) | [json](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log.json)|
|[slowonly_r50_8x8x1_196e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_8x8x1_196e_kinetics400_flow.py) |8x4| ResNet50 | ImageNet |65.76|86.25|x|8455| [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log) | [json](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log.json)|
|[slowonly_r50_video_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|x| ResNet50  | None |x|x|x|x| [ckpt](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/) | [log](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/)| [json](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmaction/)|

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

Example: train SlowOnly model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    --work_dir work_dirs/slowonly_r50_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average_clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
