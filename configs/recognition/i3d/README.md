# I3D

## Model Zoo

### Kinetics-400

|config | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |8| ResNet50|ImageNet |72.68|90.78|1.7 (320x3 frames)| 5170|[ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log.json)|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) |8x2| ResNet50| ImageNet|72.77|90.57|1.7 (320x3 frames)| 5170| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_32x2x1_100e_kinetics400_rgb_20200616-2bbb4361.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log.json)|
|[i3d_r50_fast_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb.py) |8| ResNet50 |ImageNet|72.32|90.72|1.8 (320x3 frames)| 5170| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/i3d_r50_fast_32x2x1_100e_kinetics400_rgb_20200612-000e4d2a.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log.json)|

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

Example: train I3D model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    --work_dir work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
