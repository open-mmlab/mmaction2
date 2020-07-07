# TSN

## Model Zoo

### UCF-101

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_100e_ucf101_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_80e_ucf101_rgb.py) |x| ResNet50 | ImageNet |80.12|96.09|8332| [ckpt]() | [log]()| [json]()|

### Kinetics-400

|config | gpus | backbone|pretrain | top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |x| ResNet50 | ImageNet|70.60|89.26|4.3 (25x10 frames)|8344| [ckpt]() | [log]()| [json]()|
|[tsn_r50_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x5_50e_kinetics400_rgb.py) |x| ResNet50| ImageNet |68.64|88.19|86.7 (8x1 frames)|7031| [ckpt]() | [log]()| [json]()|
|[tsn_r50_dense_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb.py) |x| ResNet50| ImageNet |68.59|88.31|12.7 (8x10 frames)|7028| [ckpt]() | [log]()| [json]()|
|[tsn_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_100e_kinetics400_rgb.py) |x| ResNet50| ImageNet |69.41|88.37|81.6 (8x1 frames)| x | [ckpt]() | [log]()| [json]()|
|[tsn_r50_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb.py) |x| ResNet50| ImageNet |70.91|89.51|10.7 (25x3 frames)| 8344 | [ckpt]() | [log]() | [json]()|
|[tsn_r50_320p_1x1x3_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow.py) |x| ResNet50 | ImageNet|55.70|79.85|x| 8471 | [ckpt]() | [log]() | [json]()|
|tsn_r50_320p_1x1x3_kinetics400_twostream [1: 1]* |x| ResNet50 | ImageNet|72.76|90.52| x | x | [ckpt]() | [log]()  | [json]()|
|[tsn_r50_320p_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py) |x| ResNet50| ImageNet |72.41|90.55|11.1 (25x3 frames)| 8344  | [ckpt]() | [log]() | [json]()|
|[tsn_r50_320p_1x1x8_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow.py) |x| ResNet50 | ImageNet|57.76|80.99|x| 8473 | [ckpt]() | [log]() | [json]()|
|tsn_r50_320p_1x1x8_kinetics400_twostream [1: 1]* |x| ResNet50| ImageNet |74.64|91.77| x | x | [ckpt]() | [log]()  | [json]()|
|[tsn_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb.py) |x| ResNet50 | ImageNet|70.77|89.3|12.2 (8x10 frames)|8344| [ckpt]() | [log]()| [json]()|
|[tsn_r50_video_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py) |x| ResNet50| ImageNet | x | x |8339| [ckpt]() | [log]()| [json]()|

*We combine rgb and flow score with coefficients 1: 1 to get the two-stream prediction (without applying softmax).

### Something-Something V1

|config | gpus| backbone |pretrain| top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py) |x| ResNet50 | ImageNet|18.55|44.80| 10978 | [ckpt]() | [log]()| [json]()|
|[tsn_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb.py) |x| ResNet50| ImageNet |15.77|39.85| 5691 | [ckpt]() | [log]()| [json]()|

### Something-Something V2

|config | gpus| backbone| pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb.py) |x| ResNet50| ImageNet |32.41|64.05| 10978 | [ckpt]() | [log]()| [json]()|
|[tsn_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py) |x| ResNet50| ImageNet |22.48|49.08|5698| [ckpt]() | [log]()| [json]()|

### Moments in Time

|config | gpus| backbone | pretrain | top1 acc| top5 acc | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x6_100e_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb.py) |x| ResNet50| ImageNet |26.84|51.6| 8339| [ckpt]() | [log]()| [json]()|

### Multi-Moments in Time

|config | gpus| backbone | pretrain | mAP| gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r101_1x1x5_50e_mmit_rgb](/configs/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb.py) |x| ResNet101| ImageNet |61.09| 10467 | [ckpt]() | [log]()| [json]()|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.

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
    --work_dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md).

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

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md).
