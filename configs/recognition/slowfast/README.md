# SlowFast

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py) |short-side 320|7x3| ResNet50|None |75.3|92.2|1.6 ((32+4)x10x3 frames)|6203|[ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16x1_256e_kinetics400_rgb_20200618-9a124260.pth)| [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/sf_4x16.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb/slowfast_r50_4x16_75.3_92.2.log.json)|
|[slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py) |short-side 320|8x4| ResNet50 |None|76.36|92.56|1.3 ((32+8)x10x3 frames)|9062| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200619-ecd36535.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/sf_8x8.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8_76.36_92.56.log.json)|

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

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowfast_r50_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
