# TSM

[TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The explosive growth in video streaming gives rise to challenges on performing video understanding at high accuracy and low computation cost. Conventional 2D CNNs are computationally cheap but cannot capture temporal relationships; 3D CNN based methods can achieve good performance but are computationally intensive, making it expensive to deploy. In this paper, we propose a generic and effective Temporal Shift Module (TSM) that enjoys both high efficiency and high performance. Specifically, it can achieve the performance of 3D CNN but maintain 2D CNN's complexity. TSM shifts part of the channels along the temporal dimension; thus facilitate information exchanged among neighboring frames. It can be inserted into 2D CNNs to achieve temporal modeling at zero computation and zero parameters. We also extended TSM to online setting, which enables real-time low-latency online video recognition and video object detection. TSM is accurate and efficient: it ranks the first place on the Something-Something leaderboard upon publication; on Jetson Nano and Galaxy Note8, it achieves a low latency of 13ms and 35ms for online video recognition.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019083-abc0de39-9ea1-4175-be5c-073c90de64c3.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

|config | resolution | gpus | backbone | pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |70.24|89.56|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7079 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log.json)|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.59|89.52|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/20200725_031623.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/20200725_031623.log.json)|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |short-side 320|8| ResNet50| ImageNet |70.73|89.81|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20210701-68d582b4.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20210616_021451.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20210616_021451.log.json)|
|[tsm_r50_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb.py) |short-side 320|8| ResNet50| ImageNet |71.90|90.03|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb/tsm_r50_1x1x8_100e_kinetics400_rgb_20210701-7ff22268.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb/20210617_103543.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_100e_kinetics400_rgb/20210617_103543.log.json)|
|[tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb.py](/configs/recognition/tsm/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.48|89.40|x|x|x|7076|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb_20210219-bf96e6cc.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb_20210219.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb/tsm_r50_gpu_normalize_1x1x8_50e_kinetics400_rgb_20210219.json)|
|[tsm_r50_video_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.25|89.66|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7077 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log.json)|
|[tsm_r50_dense_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_50e_kinetics400_rgb.py) |short-side 320|8| ResNet50 | ImageNet|73.46|90.84|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_50e_kinetics400_rgb/tsm_r50_dense_1x1x8_50e_kinetics400_rgb_20210701-a54ff3d3.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_50e_kinetics400_rgb/20210617_103245.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_50e_kinetics400_rgb/20210617_103245.log.json)|
|[tsm_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py) |short-side 320|8| ResNet50 | ImageNet|74.55|91.74|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/tsm_r50_dense_1x1x8_100e_kinetics400_rgb_20210701-e3e5e97f.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20210613_034931.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20210613_034931.log.json)|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |72.09|90.37|[70.67](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|[89.98](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|47.0 (16x1 frames)| 10404  | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/tsm_r50_340x256_1x1x16_50e_kinetics400_rgb_20201011-2f27f229.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20201011_205356.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20201011_205356.log.json)|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) |short-side 256|8x4| ResNet50| ImageNet |71.89|90.73|x|x|x|10398|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/tsm_r50_256p_1x1x16_50e_kinetics400_rgb_20201010-85645c2a.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/20201010_224825.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/20201010_224825.log.json)|
|[tsm_r50_1x1x16_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_100e_kinetics400_rgb.py) |short-side 320|8| ResNet50| ImageNet |72.80|90.75|x|x|x|10398|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_100e_kinetics400_rgb/tsm_r50_1x1x16_100e_kinetics400_rgb_20210701-41ac92b9.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_100e_kinetics400_rgb/20210618_193859.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_100e_kinetics400_rgb/20210618_193859.log.json)|
|[tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4| ResNet50| ImageNet |72.03|90.25|71.81|90.36|x|8931|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb_20200724-f00f1336.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200724_120023.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200724_120023.log.json)|
|[tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4| ResNet50| ImageNet |70.70|89.90|x|x|x|10125|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb_20200816-b93fd297.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200815_210253.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200815_210253.log.json)|
|[tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4|ResNet50| ImageNet |71.60|90.34|x|x|x|8358|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/20200723_220442.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/20200723_220442.log.json)|
|[tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb.py)|short-side 320|8|MobileNetV2| ImageNet |68.46|88.64|x|x|x|3385|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb/tsm_mobilenetv2_dense_320p_1x1x8_100e_kinetics400_rgb_20210202-61135809.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb/20210129_024936.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb/20210129_024936.log.json)|
|[tsm_mobilenetv2_dense_1x1x8_kinetics400_rgb_port](/configs/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_100e_kinetics400_rgb.py)|short-side 320|8|MobileNetV2| ImageNet |69.89|89.01|x|x|x|3385|[infer_ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_mobilenetv2_dense_1x1x8_kinetics400_rgb_port_20210922-aa5cadf6.pth)|x|x|

### Diving48

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_video_1x1x8_50e_diving48_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x8_50e_diving48_rgb.py)| 8 | ResNet50 | ImageNet | 75.99 | 97.16 | 7070 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_50e_diving48_rgb/tsm_r50_video_1x1x8_50e_diving48_rgb_20210426-aba5aa3d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_50e_diving48_rgb/20210426_012424.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_50e_diving48_rgb/20210426_012424.log.json)|
|[tsm_r50_video_1x1x16_50e_diving48_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x16_50e_diving48_rgb.py)| 8 | ResNet50 | ImageNet | 81.62 | 97.66 | 7070 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x16_50e_diving48_rgb/tsm_r50_video_1x1x16_50e_diving48_rgb_20210426-aa9631c0.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x16_50e_diving48_rgb/20210426_012823.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x16_50e_diving48_rgb/20210426_012823.log.json)|

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| reference top1 acc (efficient/accurate)| reference top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 45.58 / 47.70|75.02 / 76.12|[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/tsm_r50_1x1x8_50e_sthv1_rgb_20210203-01dce462.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20210203_150227.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20210203_150227.log.json)|
|[tsm_r50_flip_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_flip_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 47.10 / 48.51|76.02 / 77.56|[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_1x1x8_50e_sthv1_rgb/tsm_r50_flip_1x1x8_50e_sthv1_rgb_20210203-12596f16.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_1x1x8_50e_sthv1_rgb/20210203_145829.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_1x1x8_50e_sthv1_rgb/20210203_145829.log.json)|
|[tsm_r50_randaugment_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 47.16 / 48.90|76.07 / 77.92|[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb_20210324-481268d9.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_randaugment_1x1x8_50e_sthv1_rgb.json)|
|[tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 47.65 / 48.66 | 76.67 / 77.41 |[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb-ee93e5e3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_randaugment_1x1x8_50e_sthv1_rgb.json) |
|[tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 46.26 / 47.68 | 75.92 / 76.49 |[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb-4f4f4740.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb/tsm_r50_ptv_augmix_1x1x8_50e_sthv1_rgb.json) |
|[tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet| 47.85 / 50.31|76.78 / 78.18|[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb_20210324-76937692.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb/tsm_r50_flip_randaugment_1x1x8_50e_sthv1_rgb.json)|
|[tsm_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb.py)|height 100|8| ResNet50 | ImageNet|47.77 / 49.03|76.82 / 77.83|[47.05 / 48.61](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[76.40 / 77.96](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|10390|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/tsm_r50_1x1x16_50e_sthv1_rgb_20211202-b922e5d2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/tsm_r50_1x1x16_50e_sthv1_rgb.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/tsm_r50_1x1x16_50e_sthv1_rgb.json)|
|[tsm_r101_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb.py)|height 100|8| ResNet50 | ImageNet|46.09 / 48.59|75.41 / 77.10|[46.64 / 48.13](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[75.40 / 77.31](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|9800|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/tsm_r101_1x1x8_50e_sthv1_rgb_20211202-49970a5b.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/tsm_r101_1x1x8_50e_sthv1_rgb.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/tsm_r101_1x1x8_50e_sthv1_rgb.json)|

### Something-Something V2

|config | resolution | gpus | backbone | pretrain| top1 acc (efficient/accurate)| top5 acc (efficient/accurate)|  reference top1 acc (efficient/accurate)| reference top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py) |height 256|8| ResNet50| ImageNet |59.11 / 61.82|85.39 / 86.80|[xx / 61.2](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[xx / xx](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7069 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/tsm_r50_256h_1x1x8_50e_sthv2_rgb_20210816-032aa4da.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/20210816_224310.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/20210816_224310.log.json)|
|[tsm_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py) |height 256|8| ResNet50| ImageNet |61.06 / 63.19|86.66 / 87.93|[xx / 63.1](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[xx / xx](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10400 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/tsm_r50_256h_1x1x16_50e_sthv2_rgb_20210331-0a45549c.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20210331_134458.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20210331_134458.log.json)|
|[tsm_r101_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb.py) |height 256|8| ResNet101 | ImageNet|60.88 / 63.84|86.56 / 88.30|[xx / 63.3](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[xx / xx](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 9727 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/tsm_r101_256h_1x1x8_50e_sthv2_rgb_20210401-df97f3e1.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20210401_143656.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20210401_143656.log.json)|

### MixUp & CutMix on Something-Something V1

| config                                                       | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) | delta top1 acc (efficient/accurate) | delta top5 acc (efficient/accurate) |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :---------------------------------: | :---------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsm_r50_mixup_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_mixup_1x1x8_50e_sthv1_rgb.py) | height 100 |  8   | ResNet50 | ImageNet |         46.35 / 48.49         |         75.07 / 76.88         |            +0.77 / +0.79            |            +0.05 / +0.70            | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_mixup_1x1x8_50e_sthv1_rgb/tsm_r50_mixup_1x1x8_50e_sthv1_rgb-9eca48e5.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_mixup_1x1x8_50e_sthv1_rgb/tsm_r50_mixup_1x1x8_50e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_mixup_1x1x8_50e_sthv1_rgb/tsm_r50_mixup_1x1x8_50e_sthv1_rgb.json) |
| [tsm_r50_cutmix_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb.py) | height 100 |  8   | ResNet50 | ImageNet |         45.92 / 47.46         |         75.23 / 76.71         |            +0.34 / -0.24            |            +0.21 / +0.59            | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb-34934615.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb/tsm_r50_cutmix_1x1x8_50e_sthv1_rgb.json) |

### Jester

| config                                                       | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) |                             ckpt                             |                             log                              |                             json                             |
| ------------------------------------------------------------ | :--------: | :--: | :------: | :------: | :---------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [tsm_r50_1x1x8_50e_jester_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_jester_rgb.py) | height 100 |  8   | ResNet50 | ImageNet |          96.5 / 97.2          | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_jester_rgb/tsm_r50_1x1x8_50e_jester_rgb-c799267e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_jester_rgb/tsm_r50_1x1x8_50e_jester_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_jester_rgb/tsm_r50_1x1x8_50e_jester_rgb.json) |

### HMDB51

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb](/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb.py)|8|ResNet50|Kinetics400|72.68|92.03|10388|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_20210630-10c74ee5.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb/20210605_182554.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb/20210605_182554.log.json)|
|[tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb](/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb.py)|8|ResNet50|Kinetics400|74.77|93.86|10388|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb_20210630-4785548e.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb/20210605_182505.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_hmdb51_rgb/20210605_182505.log.json)|

### UCF101

|config | gpus | backbone | pretrain | top1 acc| top5 acc | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb](/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb.py)|8|ResNet50|Kinetics400|94.50|99.58|10389|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb_20210630-1fae312b.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/20210605_182720.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_ucf101_rgb/20210605_182720.log.json)|
|[tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb](/configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb.py)|8|ResNet50|Kinetics400|94.58|99.37|10389|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb_20210630-8df9c358.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/20210605_182720.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_k400_pretrained_r50_1x1x16_25e_ucf101_rgb/20210605_182720.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings. The checkpoints for reference repo can be downloaded [here](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_reference_ckpt.rar).
4. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip), which is referred from the [original repo](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd).
   We use efficient setting as default provided in config files, and it can be changed to accurate setting by

```python
...
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,   # `num_clips = 8` when using 8 segments
        twice_sample=True,    # set `twice_sample=True` for twice sample in accurate setting
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224), it is used for efficient setting
    dict(type='ThreeCrop', crop_size=256),  # it is used for accurate setting
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
```

5. When applying Mixup and CutMix, we use the hyper parameter `alpha=0.2`.
6. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
7. The **infer_ckpt** means those checkpoints are ported from [TSM](https://github.com/mit-han-lab/temporal-shift-module/blob/master/test_models.py).

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](/docs/data_preparation.md).

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

## Citation

```BibTeX
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

<!-- [BACKBONE] -->

```BibTeX
@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```
