# TSN

## Model Zoo

### UCF-101

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_80e_ucf101_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb.py) |8| ResNet50 | ImageNet |80.12|96.09|8332| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb/tsn_r50_1x1x3_80e_ucf101_rgb_20200613-d6ad9c48.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb/20200613_020013.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb/20200613_020013.log.json)|

### Kinetics-400

|config | resolution | gpus | backbone|pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.60|89.26|x|x|4.3 (25x10 frames)|8344| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json)|
|[tsn_r50_dense_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb.py) |340x256|8x3| ResNet50| ImageNet |70.18|89.10|[69.15](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.56](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.7 (8x10 frames)|7028| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/tsn_r50_dense_1x1x5_100e_kinetics400_rgb_20200627-a063165f.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log.json)|
|[tsn_r50_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb.py) |short-side 320|8x2| ResNet50| ImageNet |70.91|89.51|x|x|10.7 (25x3 frames)| 8344 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log) | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log.json)|
|[tsn_r50_320p_1x1x3_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow.py) |short-side 320|8x2| ResNet50 | ImageNet|55.70|79.85|x|x|x| 8471 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_320p_1x1x3_110e_kinetics400_flow_20200705-3036bab6.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log) | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log.json)|
|tsn_r50_320p_1x1x3_kinetics400_twostream [1: 1]* |x|x| ResNet50 | ImageNet|72.76|90.52| x | x | x | x  | x|x|x|
|[tsn_r50_320p_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py) |short-side 320|8x3| ResNet50| ImageNet |72.41|90.55|x|x|11.1 (25x3 frames)| 8344  | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log) | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log.json)|
|[tsn_r50_320p_1x1x8_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow.py) |short-side 320|8x4| ResNet50 | ImageNet|57.76|80.99|x|x|x| 8473 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_320p_1x1x8_110e_kinetics400_flow_20200705-1f39486b.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log)  | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log.json)|
|tsn_r50_320p_1x1x8_kinetics400_twostream [1: 1]* |x|x| ResNet50| ImageNet |74.64|91.77| x | x | x | x | x|x|x|
|[tsn_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.77|89.3|[68.75](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.42](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.2 (8x10 frames)|8344| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_dense_1x1x8_100e_kinetics400_rgb_20200606-e925e6e3.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log.json)|
|[tsn_r50_video_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet | 71.79 | 90.25 |x|x|x|21558| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log.json)|
|[tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet | 70.4 | 89.12 |x|x|x|21553| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb_20200703-0f19175f.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log.json)|

Here, We use [1: 1] to indicate that we combine rgb and flow score with coefficients 1: 1 to get the two-stream prediction (without applying softmax).

### Something-Something V1

|config|resolution | gpus| backbone |pretrain| top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py)|height 100 |8| ResNet50 | ImageNet|18.55 |44.80 |[17.53](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[44.29](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10978 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_1x1x8_50e_sthv1_rgb_20200618-061b9195.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_sthv1.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_f8_sthv1_18.1_45.0.log.json)|
|[tsn_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb.py)| height 100 |8| ResNet50| ImageNet |15.77 |39.85 |[13.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[35.58](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 5691 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/tsn_r50_1x1x16_50e_sthv1_rgb_20200614-7e2fe4f1.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log.json)|

### Something-Something V2

|config |resolution| gpus| backbone| pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb.py)|height 240 |8x2| ResNet50| ImageNet |32.41 |64.05 |[30.32](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[58.38](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10978 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/tsn_r50_1x1x8_50e_sthv2_rgb_20200618-096db436.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/tsn_sthv2.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/tsn_r50_f8_sthv2_32.4_64.1.log.json)|
|[tsn_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py)| height 240 |8|ResNet50| ImageNet |22.48 |49.08 |[22.50](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[47.29](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|5698| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20200614-b55c5700.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20200614_203248.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20200614_203248.log.json)|

### Moments in Time

|config |resolution| gpus| backbone | pretrain | top1 acc| top5 acc | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x6_100e_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb.py)|short-side 256 |8x2| ResNet50| ImageNet |26.84|51.6| 8339| [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_1x1x6_100e_mit_rgb_20200618-d512ab1b.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_mit.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_f6_mit_26.8_51.6.log.json)|

### Multi-Moments in Time

|config | resolution|gpus| backbone | pretrain | mAP| gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r101_1x1x5_50e_mmit_rgb](/configs/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb.py)|short-side 256 |8x2| ResNet101| ImageNet |61.09| 10467 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_1x1x5_50e_mmit_rgb_20200618-642f450d.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_mmit.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_f6_mmit_61.1.log.json)|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.

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
    --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

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

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset#test-a-dataset).
