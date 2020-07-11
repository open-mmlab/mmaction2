# Benchmark

We compare our results with some popular frameworks and official releases in terms of speed.

## Comparision Rules

Here we compare our MMAction2 repo with other video understanding toolboxes in the same data and model settings
by the training time per iteration. Here, we use
- commit id [7f3490d](https://github.com/open-mmlab/mmaction/tree/7f3490d3db6a67fe7b87bfef238b757403b670e3)(1/5/2020) of MMAction V0.1
- commit id [8d53d6f](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd)(5/5/2020) of Temporal-Shift-Module
- commit id [133e40f](https://github.com/facebookresearch/SlowFast/tree/133e40f8349ce37b0e6168639da0811a413579c8)(30/5/2020) of PySlowFast
- commit id [f13707f](https://github.com/wzmsltw/BSN-boundary-sensitive-network/tree/f13707fbc362486e93178c39f9c4d398afe2cb2f)(12/12/2018) of BSN(boundary sensitive network)
- commit id [45d0514](https://github.com/JJBOY/BMN-Boundary-Matching-Network/tree/45d05146822b85ca672b65f3d030509583d0135a)(17/10/2019) of BMN(boundary matching network)

To ensure the fairness of the comparison, the comparison experiments were conducted under the same hardware environment and using the same dataset.
For each model setting, we kept the same data preprocessing methods to make sure the same feature input.
In addition, we also used MemCache, a distributed cached system, to load the data for the same IO time except for fair comparisons with Pyslowfast which uses raw videos directly from disk.

The time we measured is the average training time for an iteration, including data processing and model training.
The training speed is measure with s/iter. The lower, the better.

## Recognizers

| Model | MMAction2 (s/iter) | MMAction V0.1 (s/iter) | Temporal-Shift-Module (s/iter) | PySlowFast (s/iter) |
| :--- | :---------------: | :--------------------: | :----------------------------: | :-----------------: |
| TSN ([tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py))   | **0.29** | 0.36 | 0.45 | x |
| I3D ([i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)) | **0.45** | 0.58 | x | x |
| I3D ([i3d_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_8x8x1_100e_kinetics400_rgb.py))   | **0.32** | x | x | 0.56 |
| TSM ([tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py))     | **0.30** | x | 0.38 | x |
| Slowonly ([slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)) | **0.30** | x | x | 1.03 |
| Slowonly ([slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py))   | **0.50** | x | x | 1.29 |
| Slowfast ([slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py)) | **0.80** | x | x | 1.40 |
| Slowfast ([slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py))   | **1.05** | x | x | 1.41 |
| R(2+1)D ([r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py))    | **0.48** | x | x | x |

## Localizers

| Model | MMAction2 (s/iter) | BSN(boundary sensitive network) (s/iter) |BMN(boundary matching network) (s/iter)|
| :--- | :---------------: | :-------------------------------------: | :-------------------------------------: |
| BSN ([TEM + PEM + PGM](/configs/localization/bsn)) | **0.074(TEM)+0.040(PEM)** | 0.101(TEM)+0.040(PEM) | x |
| BMN ([bmn_400x100_2x8_9e_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py)) | **3.27** | x | 3.30 |

## Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

## Software Environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08
