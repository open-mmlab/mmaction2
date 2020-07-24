# TSM

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone | pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |70.24|89.56|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7079 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log.json)|
|[tsm_r50_video_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.25|89.66|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7077 | [ckpt]( https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log.json)|
|[tsm_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py) |340x256|8x4| ResNet50 | ImageNet|72.9|90.44|[72.22](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#dense-sample)|[90.37](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#dense-sample)|11.5 (8x10 frames)| 7079 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/tsm_r50_dense_1x1x8_100e_kinetics400_rgb_20200626-91a54551.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20200626_213415.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20200626_213415.log.json)|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |71.69|90.4|[70.67](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|[89.98](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|47.0 (16x1 frames)| 10404  | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/tsm_r50_1x1x16_50e_kinetics400_rgb_20200607-f731bffc.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20200607_221310.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20200607_221310.log.json)|

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py) |short-side 100|8| ResNet50 | ImageNet|44.62|75.51|[42.08](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[72.66](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/tsm_r50_1x1x8_50e_sthv1_rgb_20200616-3417f361.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20200616_022852.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20200616_022852.log.json)|

### Something-Something V2

|config | resolution | gpus | backbone | pretrain| top1 acc| top5 acc |  reference top1 acc | reference top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb.py) |short-side 240|8| ResNet50| ImageNet |57.68 |83.65 |[56.57](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[84.30](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10400| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/tsm_r50_1x1x16_50e_sthv2_rgb_20200621-60ff441a.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20200621_101921.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20200621_101921.log.json)|
|[tsm_r101_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb.py) |short-side 240|8| ResNet101 | ImageNet|59.12|85.74|[59.20](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[85.27](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 9784 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/tsm_r101_1x1x8_50e_sthv2_rgb_20200625-df82f5e6.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20200625_224131.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20200625_224131.log.json)|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](/docs/data_preparation.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py \
    --work-dir work_dirs/tsm_r50_1x1x8_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
