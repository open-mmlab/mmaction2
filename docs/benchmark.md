# Benchmark

We compare our results with some other popular frameworks in terms of speed.

## Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

## Software Environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

Here we compare our MMAction repo with other Video understanding toolboxes in the same data and model settings
by the training time per iteration.

## Recognizers

| Model | MMAction (s/iter) | MMAction V0.1 (s/iter) | temporal-shift-module (s/iter) | PySlowFast (s/iter) |
| :---: | :---------------: | :--------------------: | :----------------------------: | :-----------------: |
| TSN ([tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py))   | 0.29 | 0.36 | 0.45 | None |
| I3D ([i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)) | 0.45 | 0.58 | None | None |
| TSM ([tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py))     | 0.30 | None | 0.38 | None |
| I3D ([i3d_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_8x8x1_100e_kinetics400_rgb.py))   | 0.32 | None | None | 0.56 |
| Slowonly ([slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)) | 0.30 | None | None | 1.03 |
| Slowonly ([slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py))   | 0.50 | None | None | 1.29 |
| Slowfast ([slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py)) | 0.80 | None | None | 1.40 |
| Slowfast ([slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py))   | 1.05 | None | None | 1.41 |

## Localizers

| Model | MMAction (s/iter) | BSN-boundary-sensitive-network (s/iter) |
| :---: | :---------------: | :-------------------------------------: |
| BSN ([TEM + PEM + PGM](/configs/localization/bsn)) | 0.074(TEM)+0.040(PEM) | 0.101(TEM)+0.040(PEM) |
| BMN ([bmn_400x100_2x8_9e_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py)) | 3.27 | 3.30 |
