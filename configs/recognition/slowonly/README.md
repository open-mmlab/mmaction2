# SlowOnly

## Introduction

[ALGORITHM]

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)|short-side 256|8x4| ResNet50 | None |72.76|90.51|x|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json)|
|[slowonly_r50_video_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|short-side 320|8x2| ResNet50 | None |72.90|90.82|x|8472|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.json)|
|[slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) |short-side 256|8x4| ResNet50 | None |74.42|91.49|x|5820|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb_20200820-75851a7d.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log.json)|
|[slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)|short-side 320|8x2| ResNet50 | None |73.02|90.77|4.0 (40x3 frames)|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth)| [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json)|
|[slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) |short-side 320|8x3| ResNet50 | None |74.93|91.92|2.3 (80x3 frames)|5820| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/so_8x8.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8_74.93_91.92.log.json)|
|[slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb.py)|short-side 320|8x2| ResNet50 | ImageNet |73.39|91.12|x|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912-1e8fc736.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.json)|
|[slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py) |short-side 320|8x4| ResNet50 | ImageNet |75.55|92.04|x|5820|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.json)|
|[slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb.py) | short-side 320 | 8x2 | ResNet50 | ImageNet | 74.54 | 91.73 | x | 4435 |[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb_20210308-0d6e5a69.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log.json)|
|[slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb.py) | short-side 320 | 8x4 | ResNet50 | ImageNet | 76.07 | 92.42 | x | 8895 |[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_20210308-e8dd9e82.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log.json)|
|[slowonly_r50_4x16x1_256e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow.py)|short-side 320|8x2| ResNet50  | ImageNet |61.79|83.62|x|8450| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log.json)|
|[slowonly_r50_8x8x1_196e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py) |short-side 320|8x4| ResNet50 | ImageNet |65.76|86.25|x|8455| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log.json)|

### Kinetics-400 Data Benchmark

In data benchmark, we compare two different data preprocessing methods: (1) Resize video to 340x256, (2) Resize the short edge of video to 320px, (3) Resize the short edge of video to 256px.

| config                                                       |   resolution   | gpus | backbone | Input | pretrain | top1 acc | top5 acc |  testing protocol  |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :---: | :------: | :------: | :------: | :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb.py) |    340x256     | 8x2  | ResNet50 | 4x16  |   None   |  71.61   |  90.05   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803-dadca1a3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.json) |
| [slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 | 4x16  |   None   |  73.02   |  90.77   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
| [slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb.py) | short-side 256 | 8x4  | ResNet50 | 4x16  |   None   |  72.76   |  90.51   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json) |

### Kinetics-400 OmniSource Experiments

|                            config                            |   resolution   | backbone  | pretrain |   w. OmniSource    | top1 acc | top5 acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :-------: | :------: | :----------------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | ResNet50  |   None   |        :x:         |   73.0   |   90.8   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
|                              x                               |       x        | ResNet50  |   None   | :heavy_check_mark: |   76.8   |   92.5   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r50_omni_4x16x1_kinetics400_rgb_20200926-51b1f7ea.pth) |                              x                               |                              x                               |
| [slowonly_r101_8x8x1_196e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r101_8x8x1_196e_kinetics400_rgb.py) |       x        | ResNet101 |   None   |        :x:         |   76.5   |   92.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_without_omni_8x8x1_kinetics400_rgb_20200926-0c730aef.pth) |                              x                               |                              x                               |
|                              x                               |       x        | ResNet101 |   None   | :heavy_check_mark: |   80.4   |   94.4   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth) |                              x                               |                              x                               |

### Kinetics-600

| config                                                       |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics600_rgb](/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |   77.5   |   93.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015-81e5153e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.json) |

### Kinetics-700

| config                                                       |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics700_rgb](/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |   65.0   |   86.1   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015-9250f662.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.json) |

### GYM99

| config                                                       |   resolution   | gpus | backbone | pretrain | top1 acc | mean class acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb.py) | short-side 256 | 8x2  | ResNet50 | ImageNet |   79.3   |      70.2      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111-a9c34b54.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.json) |
| [slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow](/configs/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow.py) | short-side 256 | 8x2  | ResNet50 | Kinetics |   80.3   |      71.0      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111-66ecdb3c.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.json) |
| 1: 1 Fusion                                                  |                |      |          |          |   83.7   |      74.8      |                                                              |                                                              |                                                              |

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
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
    --work-dir work_dirs/slowonly_r50_4x16x1_256e_kinetics400_rgb \
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
    --out result.json --average-clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
