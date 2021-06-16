# TSN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{wang2016temporal,
  title={Temporal segment networks: Towards good practices for deep action recognition},
  author={Wang, Limin and Xiong, Yuanjun and Wang, Zhe and Qiao, Yu and Lin, Dahua and Tang, Xiaoou and Van Gool, Luc},
  booktitle={European conference on computer vision},
  pages={20--36},
  year={2016},
  organization={Springer}
}
```

## 模型库

### UCF-101

|配置文件 | GPU 数量 | 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_75e_ucf101_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py) [1] |8| ResNet50 | ImageNet |83.03|96.78|8332| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023.json) |

[1] 这里汇报的是 UCF-101 的 split1 部分的结果。

### Diving48

|配置文件 | GPU 数量 | 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_video_1x1x8_100e_diving48_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb.py)|8| ResNet50 | ImageNet | 71.27 | 95.74 | 5699 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/tsn_r50_video_1x1x8_100e_diving48_rgb_20210426-6dde0185.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/20210426_014138.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/20210426_014138.log.json)|
|[tsn_r50_video_1x1x16_100e_diving48_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb.py)|8| ResNet50 | ImageNet | 76.75 | 96.95 | 5705 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/tsn_r50_video_1x1x16_100e_diving48_rgb_20210426-63c5f2f7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/20210426_014103.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/20210426_014103.log.json)|

### HMDB51

|配置文件 | GPU 数量 | 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb.py)|8| ResNet50 | ImageNet | 48.95| 80.19| 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_20201123-ce6c27ed.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/20201025_231108.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/20201025_231108.log.json) |
|[tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb.py) |8| ResNet50 | Kinetics400 | 56.08 | 84.31 | 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb_20201123-7f84701b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/20201108_190805.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/20201108_190805.log.json) |
|[tsn_r50_1x1x8_50e_hmdb51_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb.py) |8| ResNet50 | Moments | 54.25 | 83.86| 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/tsn_r50_1x1x8_50e_hmdb51_mit_rgb_20201123-01526d41.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/20201112_170135.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/20201112_170135.log.json) |

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络|预训练 | top1 准确率| top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.60|89.26|x|x|4.3 (25x10 frames)|8344| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json)|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |短边 256|8| ResNet50 | ImageNet|70.42|89.03|x|x|x|8343|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log.json)|
|[tsn_r50_dense_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb.py) |340x256|8x3| ResNet50| ImageNet |70.18|89.10|[69.15](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.56](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.7 (8x10 frames)|7028| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/tsn_r50_dense_1x1x5_100e_kinetics400_rgb_20200627-a063165f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log.json)|
|[tsn_r50_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb.py) |短边 320|8x2| ResNet50| ImageNet |70.91|89.51|x|x|10.7 (25x3 frames)| 8344 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log.json)|
|[tsn_r50_320p_1x1x3_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow.py) |短边 320|8x2| ResNet50 | ImageNet|55.70|79.85|x|x|x| 8471 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_320p_1x1x3_110e_kinetics400_flow_20200705-3036bab6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log.json)|
|tsn_r50_320p_1x1x3_kinetics400_twostream [1: 1]* |x|x| ResNet50 | ImageNet|72.76|90.52| x | x | x | x  | x|x|x|
|[tsn_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py)|短边 256|8| ResNet50| ImageNet |71.80|90.17|x|x|x|8343|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/20200815_173413.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/20200815_173413.log.json)|
|[tsn_r50_320p_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py) |短边 320|8x3| ResNet50| ImageNet |72.41|90.55|x|x|11.1 (25x3 frames)| 8344  | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log.json)|
|[tsn_r50_320p_1x1x8_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow.py) |短边 320|8x4| ResNet50 | ImageNet|57.76|80.99|x|x|x| 8473 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_320p_1x1x8_110e_kinetics400_flow_20200705-1f39486b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log)  | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log.json)|
|tsn_r50_320p_1x1x8_kinetics400_twostream [1: 1]* |x|x| ResNet50| ImageNet |74.64|91.77| x | x | x | x | x|x|x|
|[tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb.py) |短边 320|8| ResNet50 | ImageNet |71.11|90.04| x | x | x | 8343 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth) |[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014.json)|
|[tsn_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.77|89.3|[68.75](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.42](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.2 (8x10 frames)|8344| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_dense_1x1x8_100e_kinetics400_rgb_20200606-e925e6e3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log.json)|
|[tsn_r50_video_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py) |短边 256|8| ResNet50| ImageNet | 71.79 | 90.25 |x|x|x|21558| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log.json)|
|[tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb.py) |短边 256|8| ResNet50| ImageNet | 70.40 | 89.12 |x|x|x|21553| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb_20200703-0f19175f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log.json)|

这里，MMAction2 使用 [1: 1] 表示以 1: 1 的比例融合 RGB 和光流两分支的融合结果（融合前不经过 softmax）

### 在 TSN 模型中使用第三方的主干网络

用户可在 MMAction2 的框架中使用第三方的主干网络训练 TSN，例如：

- [x] MMClassification 中的主干网络
- [x] TorchVision 中的主干网络
- [x] pytorch-image-models(timm) 中的主干网络

|                            配置文件                            |   分辨率   | GPU 数量 |                           主干网络                           | 预训练 | top1 准确率 | top5 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :--: | :----------------------------------------------------------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.py) | 短边 320 | 8x2  | ResNeXt101-32x4d [[MMCls](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)] | ImageNet |  73.43   |  91.01   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb-16a8b561.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.json) |
| [tsn_dense161_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.py) | 短边 320 | 8x2  | Densenet-161 [[TorchVision](https://github.com/pytorch/vision/)] | ImageNet |  72.78   |  90.75   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb-cbe85332.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.json) |
| [tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 |    8     | Swin Transformer Base [[timm](https://github.com/rwightman/pytorch-image-models)] | ImageNet |    77.51    |    92.92    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb-805380f6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.json) |

1. 由于多种原因，TIMM 中的一些模型未能收到支持，详情请参考 [PR #880](https://github.com/open-mmlab/mmaction2/pull/880)。

### Kinetics-400 数据基准测试 (8 块 GPU, ResNet50, ImageNet 预训练; 3 个视频段)

在数据基准测试中，比较：

1. 不同的数据预处理方法：(1) 视频分辨率为 340x256, (2) 视频分辨率为短边 320px, (3) 视频分辨率为短边 256px;
2. 不同的数据增强方法：(1) MultiScaleCrop, (2) RandomResizedCrop;
3. 不同的测试方法：(1) 25 帧 x 10 裁剪片段, (2) 25 frames x 3 裁剪片段.

|                            配置文件                            |   分辨率   | 训练时的数据增强 | 测试时的策略 | top1 准确率 | top5 准确率 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :-------------------: | :--------------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_multiscalecrop_340x256_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_340x256_1x1x3_100e_kinetics400_rgb.py) |    340x256     |    MultiScaleCrop     |   25x10 frames   |  70.60   |  89.26   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json) |
|                              x                               |    340x256     |    MultiScaleCrop     |   25x3 frames    |  70.52   |  89.39   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb.py) |    340x256     |   RandomResizedCrop   |   25x10 frames   |  70.11   |  89.01   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725-88cb325a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725.json) |
|                              x                               |    340x256     |   RandomResizedCrop   |   25x3 frames    |  69.95   |  89.02   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb.py) | 短边 320 |    MultiScaleCrop     |   25x10 frames   |  70.32   |  89.25   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725-9922802f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725.json) |
|                              x                               | 短边 320 |    MultiScaleCrop     |   25x3 frames    |  70.54   |  89.39   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_320p_1x1x3_100e_kinetics400_rgb.py) | 短边 320 |   RandomResizedCrop   |   25x10 frames   |  70.44   |  89.23   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log.json) |
|                              x                               | 短边 320 |   RandomResizedCrop   |   25x3 frames    |  70.91   |  89.51   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_multiscalecrop_256p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_256p_1x1x3_100e_kinetics400_rgb.py) | 短边 256 |    MultiScaleCrop     |   25x10 frames   |  70.42   |  89.03   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log.json)|
|                              x                               | 短边 256 |    MultiScaleCrop     |   25x3 frames    |  70.79   |  89.42   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_256p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_256p_1x1x3_100e_kinetics400_rgb.py) | 短边 256 |    RandomResizedCrop     |   25x10 frames   |  69.80   |  89.06   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb_20200817-ae7963ca.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/20200815_172601.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/20200815_172601.log.json)|
|                              x                               | 短边 256 |   RandomResizedCrop   |   25x3 frames    |  70.48   |  89.89   |                              x                               |                              x                               |                              x                               |

### Kinetics-400 OmniSource 实验

|                            配置文件                            |   分辨率   | 主干网络 | 预训练  |   w. OmniSource    | top1 准确率 | top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :------: | :-------: | :----------------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |    340x256     | ResNet50 | ImageNet  |        :x:         |   70.6   |   89.3   |   4.3 (25x10 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json) |
|                              x                               |    340x256     | ResNet50 | ImageNet  | :heavy_check_mark: |   73.6   |   91.0   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_imagenet_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-54192355.pth) |                              x                               |                              x                               |
|                              x                               | 短边 320 | ResNet50 | IG-1B [1] |        :x:         |   73.1   |   90.4   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_without_omni_1x1x3_kinetics400_rgb_20200926-c133dd49.pth) |                              x                               |                              x                               |
|                              x                               | 短边 320 | ResNet50 | IG-1B [1] | :heavy_check_mark: |   75.7   |   91.9   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-2863fed0.pth) |                              x                               |                              x                               |

[1] MMAction2 使用 [torch-hub](https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext/) 提供的 `resnet50_swsl` 预训练模型。

### Kinetics-600

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_video_1x1x8_100e_kinetics600_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb.py) | 短边 256 | 8x2  | ResNet50 | ImageNet |   74.8   |   92.3   |   11.1 (25x3 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015-4db3c461.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015.json) |

### Kinetics-700

| 配置文件                                                       |   分辨率   | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_video_1x1x8_100e_kinetics700_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb.py) | 短边 256 | 8x2  | ResNet50 | ImageNet |   61.7   |   83.6   |   11.1 (25x3 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015-e381a6c7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015.json) |

### Something-Something V1

|配置文件|分辨率 | GPU 数量| 主干网络 |预训练| top1 准确率| top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py)|height 100 |8| ResNet50 | ImageNet|18.55 |44.80 |[17.53](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[44.29](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10978 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_1x1x8_50e_sthv1_rgb_20200618-061b9195.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_sthv1.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_f8_sthv1_18.1_45.0.log.json)|
|[tsn_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb.py)| height 100 |8| ResNet50| ImageNet |15.77 |39.85 |[13.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[35.58](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 5691 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/tsn_r50_1x1x16_50e_sthv1_rgb_20200614-7e2fe4f1.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log.json)|

### Something-Something V2

|配置文件 |分辨率| GPU 数量| 主干网络| 预训练 | top1 准确率| top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb.py)|height 240 |8| ResNet50| ImageNet |32.97 |63.62 |[30.56](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[58.49](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10966 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/tsn_r50_1x1x8_50e_sthv2_rgb_20200915-f3b381a5.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/20200915_114139.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/20200915_114139.log.json)|
|[tsn_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py)| height 240 |8|ResNet50| ImageNet |27.21 |55.84 |[21.91](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[46.87](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|8337| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20200917-80bc3611.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20200917_105855.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20200917_105855.log.json)|

### Moments in Time

|配置文件 |分辨率| GPU 数量| 主干网络 | 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x6_100e_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb.py)|短边 256 |8x2| ResNet50| ImageNet |26.84|51.6| 8339| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_1x1x6_100e_mit_rgb_20200618-d512ab1b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_mit.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_f6_mit_26.8_51.6.log.json)|

### Multi-Moments in Time

|配置文件 | 分辨率|GPU 数量| 主干网络 | 预训练 | mAP| GPU 显存占用 (M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r101_1x1x5_50e_mmit_rgb](/configs/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb.py)|短边 256 |8x2| ResNet101| ImageNet |61.09| 10467 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_1x1x5_50e_mmit_rgb_20200618-642f450d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_mmit.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_f6_mmit_61.1.log.json)|

### ActivityNet v1.3

| 配置文件                                                       | 分辨率 | GPU 数量 | 主干网络 |  预训练   | top1 准确率 | top5 准确率 | GPU 显存占用 (M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :--------: | :--: | :------: | :---------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_320p_1x1x8_50e_activitynet_video_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb.py) |  短边 320  | 8x1  | ResNet50 | Kinetics400 |  73.93   |  93.44   |    5692    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20210301-7f8da0c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/20210228_223327.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/20210228_223327.log.json) |
| [tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb.py) |  短边 320  | 8x1  | ResNet50 | Kinetics400 |  76.90   |  94.47   |    5692    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/20210217_181313.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/20210217_181313.log.json) |
| [tsn_r50_320p_1x1x8_150e_activitynet_video_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow.py) |  340x256   | 8x2  | ResNet50 | Kinetics400 |  57.51   |  83.02   |    5780    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804.json) |
| [tsn_r50_320p_1x1x8_150e_activitynet_clip_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow.py) |  340x256   | 8x2  | ResNet50 | Kinetics400 |  59.51   |  82.69   |    5780    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804.json) |

### HVU

|                          配置文件[1]                           | tag 类别 |   分辨率   | GPU 数量 | 主干网络 | 预训练 | mAP  | HATNet[2] | HATNet-multi[2] |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :----------: | :------------: | :--: | :------: | :------: | :--: | :-------: | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r18_1x1x8_100e_hvu_action_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_action_rgb.py) |    action    | 短边 256 | 8x2  | ResNet18 | ImageNet | 57.5 |   51.8    |      53.5       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027-011b282b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_scene_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_scene_rgb.py) |    scene     | 短边 256 |  8   | ResNet18 | ImageNet | 55.2 |   55.8    |      57.2       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027-00e5748d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_object_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_object_rgb.py) |    object    | 短边 256 |  8   | ResNet18 | ImageNet | 45.7 |   34.2    |      35.1       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201102-24a22f30.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_event_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_event_rgb.py) |    event     | 短边 256 |  8   | ResNet18 | ImageNet | 63.7 |   38.5    |      39.8       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027-dea8cd71.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_concept_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_concept_rgb.py) |   concept    | 短边 256 |  8   | ResNet18 | ImageNet | 47.5 |   26.1    |      27.3       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027-fc1dd8e3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_attribute_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_attribute_rgb.py) |  attribute   | 短边 256 |  8   | ResNet18 | ImageNet | 46.1 |   33.6    |      34.9       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027-0b3b49d2.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027.json) |
|                              -                               |   所有 tag    | 短边 256 |  -   | ResNet18 | ImageNet | 52.6 |   40.0    |      41.3       |                              -                               |                              -                               |                              -                               |

[1] 简单起见，MMAction2 对每个 tag 类别训练特定的模型，作为 HVU 的基准模型。

[2] 这里 HATNet 和 HATNet-multi 的结果来自于 paper: [Large Scale Holistic Video Understanding](https://pages.iai.uni-bonn.de/gall_juergen/download/HVU_eccv20.pdf)。
HATNet 的时序动作候选是一个双分支的卷积网络（一个 2D 分支，一个 3D 分支），并且和 MMAction2 有相同的主干网络（ResNet18）。HATNet 的输入是 16 帧或 32 帧的长视频片段（这样的片段比 MMAction2 使用的要长），同时输入分辨率更粗糙（112px 而非 224px）。
HATNet 是在每个独立的任务（对应每个 tag 类别）上进行训练的，HATNet-multi 是在多个任务上进行训练的。由于目前没有 HATNet 的开源代码和模型，这里仅汇报了原 paper 的精度。

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。
4. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考：

- [准备 ucf101](/tools/data/ucf101/README_zh-CN.md)
- [准备 kinetics](/tools/data/kinetics/README_zh-CN.md)
- [准备 sthv1](/tools/data/sthv1/README_zh-CN.md)
- [准备 sthv2](/tools/data/sthv2/README_zh-CN.md)
- [准备 mit](/tools/data/mit/README_zh-CN.md)
- [准备 mmit](/tools/data/mmit/README_zh-CN.md)
- [准备 hvu](/tools/data/hvu/README_zh-CN.md)
- [准备 hmdb51](/tools/data/hmdb51/README_zh-CN.md)

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TSN 模型在 Kinetics-400 数据集上的训练。

```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 TSN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
