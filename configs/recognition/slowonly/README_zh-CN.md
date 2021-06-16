# SlowOnly

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 |预训练| top1 准确率| top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)|短边 256|8x4| ResNet50 | None |72.76|90.51|x|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json)|
|[slowonly_r50_video_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|短边 320|8x2| ResNet50 | None |72.90|90.82|x|8472|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.json)|
|[slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) |短边 256|8x4| ResNet50 | None |74.42|91.49|x|5820|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb_20200820-75851a7d.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log.json)|
|[slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)|短边 320|8x2| ResNet50 | None |73.02|90.77|4.0 (40x3 frames)|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth)| [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json)|
|[slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) |短边 320|8x3| ResNet50 | None |74.93|91.92|2.3 (80x3 frames)|5820| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/so_8x8.log)| [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8_74.93_91.92.log.json)|
|[slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb.py)|短边 320|8x2| ResNet50 | ImageNet |73.39|91.12|x|3168|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912-1e8fc736.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.json)|
|[slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py) |短边 320|8x4| ResNet50 | ImageNet |75.55|92.04|x|5820|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.json)|
|[slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb.py) | 短边 320 | 8x2 | ResNet50 | ImageNet | 74.54 | 91.73 | x | 4435 |[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb_20210308-0d6e5a69.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log.json)|
|[slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb.py) | 短边 320 | 8x4 | ResNet50 | ImageNet | 76.07 | 92.42 | x | 8895 |[ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_20210308-e8dd9e82.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log)|[json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log.json)|
|[slowonly_r50_4x16x1_256e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow.py)|短边 320|8x2| ResNet50  | ImageNet |61.79|83.62|x|8450| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log.json)|
|[slowonly_r50_8x8x1_196e_kinetics400_flow](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py) |短边 320|8x4| ResNet50 | ImageNet |65.76|86.25|x|8455| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log.json)|

### Kinetics-400 数据基准测试

在数据基准测试中，比较两种不同的数据预处理方法 (1) 视频分辨率为 340x256, (2) 视频分辨率为短边 320px, (3) 视频分辨率为短边 256px.

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 输入 | 预训练 | top1 准确率 | top5 准确率 |  测试方案  |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :---: | :------: | :------: | :------: | :----------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb.py) |    340x256     | 8x2  | ResNet50 | 4x16  |   None   |  71.61   |  90.05   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803-dadca1a3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.json) |
| [slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb.py) | 短边 320 | 8x2  | ResNet50 | 4x16  |   None   |  73.02   |  90.77   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
| [slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb.py) | 短边 256 | 8x4  | ResNet50 | 4x16  |   None   |  72.76   |  90.51   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json) |

### Kinetics-400 OmniSource Experiments

|                            配置文件                            |   分辨率   | 主干网络  | 预训练 |   w. OmniSource    | top1 准确率 | top5 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :-------: | :------: | :----------------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | 短边 320 | ResNet50  |   None   |        :x:         |   73.0   |   90.8   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
|                              x                               |       x        | ResNet50  |   None   | :heavy_check_mark: |   76.8   |   92.5   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r50_omni_4x16x1_kinetics400_rgb_20200926-51b1f7ea.pth) |                              x                               |                              x                               |
| [slowonly_r101_8x8x1_196e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r101_8x8x1_196e_kinetics400_rgb.py) |       x        | ResNet101 |   None   |        :x:         |   76.5   |   92.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_without_omni_8x8x1_kinetics400_rgb_20200926-0c730aef.pth) |                              x                               |                              x                               |
|                              x                               |       x        | ResNet101 |   None   | :heavy_check_mark: |   80.4   |   94.4   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth) |                              x                               |                              x                               |

### Kinetics-600

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics600_rgb](/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb.py) | 短边 256 | 8x4  | ResNet50 |   None   |   77.5   |   93.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015-81e5153e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.json) |

### Kinetics-700

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics700_rgb](/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb.py) | 短边 256 | 8x4  | ResNet50 |   None   |   65.0   |   86.1   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015-9250f662.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.json) |

### GYM99

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | 类别平均准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb.py) | 短边 256 | 8x2  | ResNet50 | ImageNet |   79.3   |      70.2      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111-a9c34b54.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.json) |
| [slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow](/configs/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow.py) | 短边 256 | 8x2  | ResNet50 | Kinetics |   80.3   |      71.0      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111-66ecdb3c.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.json) |
| 1: 1 融合                                                  |                |      |          |          |   83.7   |      74.8      |                                                              |                                                              |                                                              |

### Jester

| 配置文件                                                     | 分辨率 | GPU 数量 | 主干网络 |  预训练  | top1 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :----: | :------: | :------: | :------: | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb](/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.py) | 高 100 |    8     | ResNet50 | ImageNet |    97.2     | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb-b56a5389.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 SlowOnly 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowonly_r50_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 数据集上测试 SlowOnly 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
