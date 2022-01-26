# TSN

[Temporal segment networks: Towards good practices for deep action recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Deep convolutional networks have achieved great success for visual recognition in still images. However, for action recognition in videos, the advantage over traditional methods is not so evident. This paper aims to discover the principles to design effective ConvNet architectures for action recognition in videos and learn these models given limited training samples. Our first contribution is temporal segment network (TSN), a novel framework for video-based action recognition. which is based on the idea of long-range temporal structure modeling. It combines a sparse temporal sampling strategy and video-level supervision to enable efficient and effective learning using the whole action video. The other contribution is our study on a series of good practices in learning ConvNets on video data with the help of temporal segment network. Our approach obtains the state-the-of-art performance on the datasets of HMDB51 ( 69.4%) and UCF101 (94.2%). We also visualize the learned ConvNet models, which qualitatively demonstrates the effectiveness of temporal segment network and the proposed good practices.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019237-8823045b-dfa3-45cc-a992-ee83ab9d8459.png" width="800"/>
</div>

## Results and Models

### UCF-101

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_75e_ucf101_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py) [1] |8| ResNet50 | ImageNet |83.03|96.78|8332| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023-d85ab600.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb/tsn_r50_1x1x3_75e_ucf101_rgb_20201023.json) |

[1] We report the performance on UCF-101 split1.

### Diving48

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_video_1x1x8_100e_diving48_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb.py)|8| ResNet50 | ImageNet | 71.27 | 95.74 | 5699 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/tsn_r50_video_1x1x8_100e_diving48_rgb_20210426-6dde0185.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/20210426_014138.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_diving48_rgb/20210426_014138.log.json)|
|[tsn_r50_video_1x1x16_100e_diving48_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb.py)|8| ResNet50 | ImageNet | 76.75 | 96.95 | 5705 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/tsn_r50_video_1x1x16_100e_diving48_rgb_20210426-63c5f2f7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/20210426_014103.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x16_100e_diving48_rgb/20210426_014103.log.json)|

### HMDB51

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb.py)|8| ResNet50 | ImageNet | 48.95| 80.19| 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb_20201123-ce6c27ed.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/20201025_231108.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_imagenet_rgb/20201025_231108.log.json) |
|[tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb.py) |8| ResNet50 | Kinetics400 | 56.08 | 84.31 | 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb_20201123-7f84701b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/20201108_190805.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_kinetics400_rgb/20201108_190805.log.json) |
|[tsn_r50_1x1x8_50e_hmdb51_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb.py) |8| ResNet50 | Moments | 54.25 | 83.86| 21535| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/tsn_r50_1x1x8_50e_hmdb51_mit_rgb_20201123-01526d41.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/20201112_170135.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_hmdb51_mit_rgb/20201112_170135.log.json) |

### Kinetics-400

|config | resolution | gpus | backbone|pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.60|89.26|x|x|4.3 (25x10 frames)|8344| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json)|
|[tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50 | ImageNet|70.42|89.03|x|x|x|8343|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log.json)|
|[tsn_r50_dense_1x1x5_50e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb.py) |340x256|8x3| ResNet50| ImageNet |70.18|89.10|[69.15](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.56](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.7 (8x10 frames)|7028| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/tsn_r50_dense_1x1x5_100e_kinetics400_rgb_20200627-a063165f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x5_100e_kinetics400_rgb/20200627_105310.log.json)|
|[tsn_r50_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb.py) |short-side 320|8x2| ResNet50| ImageNet |70.91|89.51|x|x|10.7 (25x3 frames)| 8344 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log.json)|
|[tsn_r50_320p_1x1x3_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow.py) |short-side 320|8x2| ResNet50 | ImageNet|55.70|79.85|x|x|x| 8471 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_320p_1x1x3_110e_kinetics400_flow_20200705-3036bab6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_110e_kinetics400_flow/tsn_r50_f3_kinetics400_flow_shortedge_55.7_79.9.log.json)|
|tsn_r50_320p_1x1x3_kinetics400_twostream [1: 1]* |x|x| ResNet50 | ImageNet|72.76|90.52| x | x | x | x  | x|x|x|
|[tsn_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py)|short-side 256|8| ResNet50| ImageNet |71.80|90.17|x|x|x|8343|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/tsn_r50_256p_1x1x8_100e_kinetics400_rgb_20200817-883baf16.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/20200815_173413.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x8_100e_kinetics400_rgb/20200815_173413.log.json)|
|[tsn_r50_320p_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb.py) |short-side 320|8x3| ResNet50| ImageNet |72.41|90.55|x|x|11.1 (25x3 frames)| 8344  | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_320p_1x1x8_100e_kinetics400_rgb_20200702-ef80e3d7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_100e_kinetics400_rgb/tsn_r50_f8_kinetics400_shortedge_72.4_90.6.log.json)|
|[tsn_r50_320p_1x1x8_110e_kinetics400_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow.py) |short-side 320|8x4| ResNet50 | ImageNet|57.76|80.99|x|x|x| 8473 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_320p_1x1x8_110e_kinetics400_flow_20200705-1f39486b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log)  | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_110e_kinetics400_flow/tsn_r50_f8_kinetics400_flow_shortedge_57.8_81.0.log.json)|
|tsn_r50_320p_1x1x8_kinetics400_twostream [1: 1]* |x|x| ResNet50| ImageNet |74.64|91.77| x | x | x | x | x|x|x|
|[tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb.py) |short-side 320|8| ResNet50 | ImageNet |71.11|90.04| x | x | x | 8343 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth) |[log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014.json)|
|[tsn_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb.py) |340x256|8| ResNet50 | ImageNet|70.77|89.3|[68.75](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[88.42](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|12.2 (8x10 frames)|8344| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_dense_1x1x8_100e_kinetics400_rgb_20200606-e925e6e3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb/20200606_003901.log.json)|
|[tsn_r50_video_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet | 71.14 | 89.63 |x|x|x|21558| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_100e_kinetics400_rgb.log.json)|
|[tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet | 70.40 | 89.12 |x|x|x|21553| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb_20200703-0f19175f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_dense_1x1x8_100e_kinetics400_rgb/tsn_r50_video_2d_1x1x8_dense_100e_kinetics400_rgb.log.json)|

Here, We use [1: 1] to indicate that we combine rgb and flow score with coefficients 1: 1 to get the two-stream prediction (without applying softmax).

### Using backbones from 3rd-party in TSN

It's possible and convenient to use a 3rd-party backbone for TSN under the framework of MMAction2, here we provide some examples for:

- [x] Backbones from [MMClassification](https://github.com/open-mmlab/mmclassification/)
- [x] Backbones from [TorchVision](https://github.com/pytorch/vision/)
- [x] Backbones from [TIMM (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models)

| config                                                       |   resolution   | gpus |                           backbone                           | pretrain | top1 acc | top5 acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :----------------------------------------------------------: | :------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNeXt101-32x4d [[MMCls](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)] | ImageNet |  73.43   |  91.01   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb-16a8b561.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.json) |
| [tsn_dense161_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 | 8x2  | Densenet-161 [[TorchVision](https://github.com/pytorch/vision/)] | ImageNet |  72.78   |  90.75   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb-cbe85332.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.json) |
| [tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 | 8  | Swin Transformer Base [[timm](https://github.com/rwightman/pytorch-image-models)] | ImageNet |  77.51  |  92.92  | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb-805380f6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.json) |

1. Note that some backbones in TIMM are not supported due to multiple reasons. Please refer to to [PR #880](https://github.com/open-mmlab/mmaction2/pull/880) for details.

### Kinetics-400 Data Benchmark (8-gpus, ResNet50, ImageNet pretrain; 3 segments)

In data benchmark, we compare:

1. Different data preprocessing methods: (1) Resize video to 340x256, (2) Resize the short edge of video to 320px, (3) Resize the short edge of video to 256px;
2. Different data augmentation methods: (1) MultiScaleCrop, (2) RandomResizedCrop;
3. Different testing protocols: (1) 25 frames x 10 crops, (2) 25 frames x 3 crops.

|                            config                            |   resolution   | training augmentation | testing protocol | top1 acc | top5 acc |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :-------------------: | :--------------: | :------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_multiscalecrop_340x256_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_340x256_1x1x3_100e_kinetics400_rgb.py) |    340x256     |    MultiScaleCrop     |   25x10 frames   |  70.60   |  89.26   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json) |
|                              x                               |    340x256     |    MultiScaleCrop     |   25x3 frames    |  70.52   |  89.39   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb.py) |    340x256     |   RandomResizedCrop   |   25x10 frames   |  70.11   |  89.01   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725-88cb325a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb/tsn_r50_randomresizedcrop_340x256_1x1x3_100e_kinetics400_rgb_20200725.json) |
|                              x                               |    340x256     |   RandomResizedCrop   |   25x3 frames    |  69.95   |  89.02   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 |    MultiScaleCrop     |   25x10 frames   |  70.32   |  89.25   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725-9922802f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_multiscalecrop_320p_1x1x3_100e_kinetics400_rgb_20200725.json) |
|                              x                               | short-side 320 |    MultiScaleCrop     |   25x3 frames    |  70.54   |  89.39   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_320p_1x1x3_100e_kinetics400_rgb.py) | short-side 320 |   RandomResizedCrop   |   25x10 frames   |  70.44   |  89.23   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_f3_kinetics400_shortedge_70.9_89.5.log.json) |
|                              x                               | short-side 320 |   RandomResizedCrop   |   25x3 frames    |  70.91   |  89.51   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_multiscalecrop_256p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_multiscalecrop_256p_1x1x3_100e_kinetics400_rgb.py) | short-side 256 |    MultiScaleCrop     |   25x10 frames   |  70.42   |  89.03   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_1x1x3_100e_kinetics400_rgb_20200725-22592236.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_1x1x3_100e_kinetics400_rgb/20200725_031325.log.json)|
|                              x                               | short-side 256 |    MultiScaleCrop     |   25x3 frames    |  70.79   |  89.42   |                              x                               |                              x                               |                              x                               |
| [tsn_r50_randomresizedcrop_256p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/data_benchmark/tsn_r50_randomresizedcrop_256p_1x1x3_100e_kinetics400_rgb.py) | short-side 256 |    RandomResizedCrop     |   25x10 frames   |  69.80   |  89.06   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb_20200817-ae7963ca.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/20200815_172601.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_256p_randomresize_1x1x3_100e_kinetics400_rgb/20200815_172601.log.json)|
|                              x                               | short-side 256 |   RandomResizedCrop   |   25x3 frames    |  70.48   |  89.89   |                              x                               |                              x                               |                              x                               |

### Kinetics-400 OmniSource Experiments

|                            config                            |   resolution   | backbone | pretrain  |   w. OmniSource    | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :------------: | :------: | :-------: | :----------------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |    340x256     | ResNet50 | ImageNet  |        :x:         |   70.6   |   89.3   |   4.3 (25x10 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log.json) |
|                              x                               |    340x256     | ResNet50 | ImageNet  | :heavy_check_mark: |   73.6   |   91.0   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_imagenet_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-54192355.pth) |                              x                               |                              x                               |
|                              x                               | short-side 320 | ResNet50 | IG-1B [1] |        :x:         |   73.1   |   90.4   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_without_omni_1x1x3_kinetics400_rgb_20200926-c133dd49.pth) |                              x                               |                              x                               |
|                              x                               | short-side 320 | ResNet50 | IG-1B [1] | :heavy_check_mark: |   75.7   |   91.9   |            x            |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-2863fed0.pth) |                              x                               |                              x                               |

[1] We obtain the pre-trained model from [torch-hub](https://pytorch.org/hub/facebookresearch_semi-supervised-ImageNet1K-models_resnext/), the pretrain model we used is `resnet50_swsl`

### Kinetics-600

| config                                                       |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_video_1x1x8_100e_kinetics600_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb.py) | short-side 256 | 8x2  | ResNet50 | ImageNet |   74.8   |   92.3   |   11.1 (25x3 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015-4db3c461.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics600_rgb/tsn_r50_video_1x1x8_100e_kinetics600_rgb_20201015.json) |

### Kinetics-700

| config                                                       |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_video_1x1x8_100e_kinetics700_rgb](/configs/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb.py) | short-side 256 | 8x2  | ResNet50 | ImageNet |   61.7   |   83.6   |   11.1 (25x3 frames)    |    8344    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015-e381a6c7.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_video_1x1x8_100e_kinetics700_rgb/tsn_r50_video_1x1x8_100e_kinetics700_rgb_20201015.json) |

### Something-Something V1

|config|resolution | gpus| backbone |pretrain| top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb.py)|height 100 |8| ResNet50 | ImageNet|18.55 |44.80 |[17.53](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[44.29](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10978 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_1x1x8_50e_sthv1_rgb_20200618-061b9195.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_sthv1.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv1_rgb/tsn_r50_f8_sthv1_18.1_45.0.log.json)|
|[tsn_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb.py)| height 100 |8| ResNet50| ImageNet |15.77 |39.85 |[13.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[35.58](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 5691 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/tsn_r50_1x1x16_50e_sthv1_rgb_20200614-7e2fe4f1.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv1_rgb/20200614_211932.log.json)|

### Something-Something V2

|config |resolution| gpus| backbone| pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb.py)|height 256 |8| ResNet50| ImageNet |28.59 |59.56 | x | x | 10966 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/tsn_r50_1x1x8_50e_sthv2_rgb_20210816-1aafee8f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/20210816_221116.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x8_50e_sthv2_rgb/20210816_221116.log.json)|
|[tsn_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb.py)| height 256 |8|ResNet50| ImageNet |20.89 |49.16 | x | x |8337| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/tsn_r50_1x1x16_50e_sthv2_rgb_20210816-5d23ac6e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20210816_225256.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x16_50e_sthv2_rgb/20210816_225256.log.json)|

### Moments in Time

|config |resolution| gpus| backbone | pretrain | top1 acc| top5 acc | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r50_1x1x6_100e_mit_rgb](/configs/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb.py)|short-side 256 |8x2| ResNet50| ImageNet |26.84|51.6| 8339| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_1x1x6_100e_mit_rgb_20200618-d512ab1b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_mit.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x6_100e_mit_rgb/tsn_r50_f6_mit_26.8_51.6.log.json)|

### Multi-Moments in Time

|config | resolution|gpus| backbone | pretrain | mAP| gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsn_r101_1x1x5_50e_mmit_rgb](/configs/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb.py)|short-side 256 |8x2| ResNet101| ImageNet |61.09| 10467 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_1x1x5_50e_mmit_rgb_20200618-642f450d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_mmit.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r101_1x1x5_50e_mmit_rgb/tsn_r101_f6_mmit_61.1.log.json)|

### ActivityNet v1.3

| config                                                       | resolution | gpus | backbone |  pretrain   | top1 acc | top5 acc | gpu_mem(M) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :--------: | :--: | :------: | :---------: | :------: | :------: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r50_320p_1x1x8_50e_activitynet_video_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb.py) |  short-side 320  | 8x1  | ResNet50 | Kinetics400 |  73.93   |  93.44   |    5692    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20210301-7f8da0c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/20210228_223327.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb/20210228_223327.log.json) |
| [tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb](/configs/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb.py) |  short-side 320  | 8x1  | ResNet50 | Kinetics400 |  76.90   |  94.47   |    5692    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb_20210301-c0f04a7e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/20210217_181313.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_50e_activitynet_clip_rgb/20210217_181313.log.json) |
| [tsn_r50_320p_1x1x8_150e_activitynet_video_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow.py) |  340x256   | 8x2  | ResNet50 | Kinetics400 |  57.51   |  83.02   |    5780    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804-13313f52.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_video_flow/tsn_r50_320p_1x1x8_150e_activitynet_video_flow_20200804.json) |
| [tsn_r50_320p_1x1x8_150e_activitynet_clip_flow](/configs/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow.py) |  340x256   | 8x2  | ResNet50 | Kinetics400 |  59.51   |  82.69   |    5780    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804-8622cf38.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow/tsn_r50_320p_1x1x8_150e_activitynet_clip_flow_20200804.json) |

### HVU

|                          config[1]                           | tag category |   resolution   | gpus | backbone | pretrain | mAP  | HATNet[2] | HATNet-multi[2] |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------: | :----------: | :------------: | :--: | :------: | :------: | :--: | :-------: | :-------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsn_r18_1x1x8_100e_hvu_action_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_action_rgb.py) |    action    | short-side 256 | 8x2  | ResNet18 | ImageNet | 57.5 |   51.8    |      53.5       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027-011b282b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/action/tsn_r18_1x1x8_100e_hvu_action_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_scene_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_scene_rgb.py) |    scene     | short-side 256 |  8   | ResNet18 | ImageNet | 55.2 |   55.8    |      57.2       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027-00e5748d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/scene/tsn_r18_1x1x8_100e_hvu_scene_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_object_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_object_rgb.py) |    object    | short-side 256 |  8   | ResNet18 | ImageNet | 45.7 |   34.2    |      35.1       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201102-24a22f30.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/object/tsn_r18_1x1x8_100e_hvu_object_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_event_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_event_rgb.py) |    event     | short-side 256 |  8   | ResNet18 | ImageNet | 63.7 |   38.5    |      39.8       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027-dea8cd71.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/event/tsn_r18_1x1x8_100e_hvu_event_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_concept_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_concept_rgb.py) |   concept    | short-side 256 |  8   | ResNet18 | ImageNet | 47.5 |   26.1    |      27.3       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027-fc1dd8e3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/concept/tsn_r18_1x1x8_100e_hvu_concept_rgb_20201027.json) |
| [tsn_r18_1x1x8_100e_hvu_attribute_rgb](/configs/recognition/tsn/hvu/tsn_r18_1x1x8_100e_hvu_attribute_rgb.py) |  attribute   | short-side 256 |  8   | ResNet18 | ImageNet | 46.1 |   33.6    |      34.9       | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027-0b3b49d2.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsn/hvu/attribute/tsn_r18_1x1x8_100e_hvu_attribute_rgb_20201027.json) |
|                              -                               |   Overall    | short-side 256 |  -   | ResNet18 | ImageNet | 52.6 |   40.0    |      41.3       |                              -                               |                              -                               |                              -                               |

[1] For simplicity, we train a specific model for each tag category as the baselines for HVU.

[2] The performance of HATNet and HATNet-multi are from the paper [Large Scale Holistic Video Understanding](https://pages.iai.uni-bonn.de/gall_juergen/download/HVU_eccv20.pdf). The proposed HATNet is a 2 branch Convolution Network (one 2D branch, one 3D branch) and share the same backbone(ResNet18) with us. The inputs of HATNet are 16 or 32 frames long video clips (which is much larger than us), while the input resolution is coarser (112 instead of 224). HATNet is trained on each individual task (each tag category) while HATNet-multi is trained on multiple tasks. Since there is no released codes or models for the HATNet, we just include the performance reported by the original paper.

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [preparing_ucf101](/tools/data/ucf101/README.md)
- [preparing_kinetics](/tools/data/kinetics/README.md)
- [preparing_sthv1](/tools/data/sthv1/README.md)
- [preparing_sthv2](/tools/data/sthv2/README.md)
- [preparing_mit](/tools/data/mit/README.md)
- [preparing_mmit](/tools/data/mmit/README.md)
- [preparing_hvu](/tools/data/hvu/README.md)
- [preparing_hmdb51](/tools/data/hmdb51/README.md)

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

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

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
