# Omni-sourced Webly-supervised Learning for Video Recognition

[Haodong Duan](https://github.com/kennymckormick), [Yue Zhao](https://github.com/zhaoyue-zephyrus), [Yuanjun Xiong](https://github.com/yjxiong), Wentao Liu, [Dahua Lin](https://github.com/lindahua)

In ECCV, 2020. [Paper](https://arxiv.org/abs/2003.13042)

![pipeline](https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/omnisource/pipeline.png?raw=true)

## Model Zoo

### Kinetics-400 Model Release

We currently released 4 models trained with OmniSource framework, including both 2D and 3D architectures. We compare the performance of models trained with or without OmniSource in the following table.

|  Model   | Modality | Pretrained | Backbone  | Input |   Resolution   | Top-1 (Baseline / OmniSource (Delta)) | Top-5 (Baseline / OmniSource (Delta))) |                           Download                           |
| :------: | :------: | :--------: | :-------: | :---: | :------------: | :-----------------------------------: | :------------------------------------: | :----------------------------------------------------------: |
|   TSN    |   RGB    |  ImageNet  | ResNet50  | 3seg  |    340x256     |          70.6 / 73.6 (+ 3.0)          |          89.4 / 91.0 (+ 1.6)           | [Baseline](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) / [OmniSource](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_imagenet_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-54192355.pth) |
|   TSN    |   RGB    |   IG-1B    | ResNet50  | 3seg  | short-side 320 |          73.1 / 75.7 (+ 2.6)          |          90.4 / 91.9 (+ 1.5)           | [Baseline](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_without_omni_1x1x3_kinetics400_rgb_20200926-c133dd49.pth) / [OmniSource](https://download.openmmlab.com/mmaction/recognition/tsn/omni/tsn_1G1B_pretrained_r50_omni_1x1x3_kinetics400_rgb_20200926-2863fed0.pth) |
| SlowOnly |   RGB    |  Scratch   | ResNet50  | 4x16  | short-side 320 |          72.9 / 76.8 (+ 3.9)          |          90.9 / 92.5 (+ 1.6)           | [Baseline](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) / [OmniSource](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r50_omni_4x16x1_kinetics400_rgb_20200926-51b1f7ea.pth) |
| SlowOnly |   RGB    |  Scratch   | ResNet101 |  8x8  | short-side 320 |          76.5 / 80.4 (+ 3.9)          |          92.7 / 94.4 (+ 1.7)           | [Baseline](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_without_omni_8x8x1_kinetics400_rgb_20200926-0c730aef.pth) / [OmniSource](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth) |

## Benchmark on Mini-Kinetics

We release a subset of web dataset used in the OmniSource paper. Specifically, we release the web data in the 200 classes of [Mini-Kinetics](https://arxiv.org/pdf/1712.04851.pdf). The statistics of those datasets is detailed in [preparing_omnisource](/tools/data/omnisource/README.md). To obtain those data, you need to fill in a [data request form](https://docs.google.com/forms/d/e/1FAIpQLSd8_GlmHzG8FcDbW-OEu__G7qLgOSYZpH-i5vYVJcu7wcb_TQ/viewform?usp=sf_link). After we received your request, the download link of these data will be send to you. For more details on the released OmniSource web dataset, please refer to [preparing_omnisource](/tools/data/omnisource/README.md).

We benchmark the OmniSource framework on the released subset, results are listed in the following table (we report the Top-1 and Top-5 accuracy on Mini-Kinetics validation). The cbenchmark can be used as a baseline for video recognition with web data.

### TSN-8seg-ResNet50

|   Setting    | Top-1 | Top-5 |                             ckpt                             |                             json                             |                             log                              |
| :----------: | :---: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   Baseline   | 77.4  | 93.6  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/baseline/tsn_r50_1x1x8_100e_minikinetics_rgb_20201030-b4eaf92b.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/baseline/tsn_r50_1x1x8_100e_minikinetics_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/baseline/tsn_r50_1x1x8_100e_minikinetics_rgb_20201030.log) |
|   +GG-img    | 78.0  | 93.6  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/googleimage/tsn_r50_1x1x8_100e_minikinetics_googleimage_rgb_20201030-23966b4b.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/googleimage/tsn_r50_1x1x8_100e_minikinetics_googleimage_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/googleimage/tsn_r50_1x1x8_100e_minikinetics_googleimage_rgb_20201030.log) |
| +[GG-IG]-img | 78.6  | 93.6  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/webimage/tsn_r50_1x1x8_100e_minikinetics_webimage_rgb_20201030-66f5e046.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/webimage/tsn_r50_1x1x8_100e_minikinetics_webimage_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/webimage/tsn_r50_1x1x8_100e_minikinetics_webimage_rgb_20201030.log) |
|   +IG-vid    | 80.6  | 95.0  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/insvideo/tsn_r50_1x1x8_100e_minikinetics_insvideo_rgb_20201030-011f984d.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/insvideo/tsn_r50_1x1x8_100e_minikinetics_insvideo_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/insvideo/tsn_r50_1x1x8_100e_minikinetics_insvideo_rgb_20201030.log) |
|    +KRaw     | 78.6  | 93.2  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/kineticsraw/tsn_r50_1x1x8_100e_minikinetics_kineticsraw_rgb_20201030-59f5d064.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/kineticsraw/tsn_r50_1x1x8_100e_minikinetics_kineticsraw_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/kineticsraw/tsn_r50_1x1x8_100e_minikinetics_kineticsraw_rgb_20201030.log) |
|  OmniSource  | 81.3  | 94.8  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/omnisource/tsn_r50_1x1x8_100e_minikinetics_omnisource_rgb_20201030-0f56ef51.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/omnisource/tsn_r50_1x1x8_100e_minikinetics_omnisource_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/tsn_r50_1x1x8_100e_minikinetics_rgb/omnisource/tsn_r50_1x1x8_100e_minikinetics_omnisource_rgb_20201030.log) |

### SlowOnly-8x8-ResNet50

|   Setting    | Top-1 | Top-5 |                             ckpt                             |                             json                             |                             log                              |
| :----------: | :---: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   Baseline   | 78.6  | 93.9  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/baseline/slowonly_r50_8x8x1_256e_minikinetics_rgb_20201030-168eb098.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/baseline/slowonly_r50_8x8x1_256e_minikinetics_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/baseline/slowonly_r50_8x8x1_256e_minikinetics_rgb_20201030.log) |
|   +GG-img    | 80.8  | 95.0  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/googleimage/slowonly_r50_8x8x1_256e_minikinetics_googleimage_rgb_20201030-7da6dfc3.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/googleimage/slowonly_r50_8x8x1_256e_minikinetics_googleimage_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/googleimage/slowonly_r50_8x8x1_256e_minikinetics_googleimage_rgb_20201030.log) |
| +[GG-IG]-img | 81.3  | 95.2  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/webimage/slowonly_r50_8x8x1_256e_minikinetics_webimage_rgb_20201030-c36616e9.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/webimage/slowonly_r50_8x8x1_256e_minikinetics_webimage_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/webimage/slowonly_r50_8x8x1_256e_minikinetics_webimage_rgb_20201030.log) |
|   +IG-vid    | 82.4  | 95.6  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/insvideo/slowonly_r50_8x8x1_256e_minikinetics_insvideo_rgb_20201030-e2890e8d.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/insvideo/slowonly_r50_8x8x1_256e_minikinetics_insvideo_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/insvideo/slowonly_r50_8x8x1_256e_minikinetics_insvideo_rgb_20201030.log) |
|    +KRaw     | 80.3  | 94.5  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/kineticsraw/slowonly_r50_8x8x1_256e_minikinetics_kineticsraw_rgb_20201030-62974bac.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/kineticsraw/slowonly_r50_8x8x1_256e_minikinetics_kineticsraw_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/kineticsraw/slowonly_r50_8x8x1_256e_minikinetics_kineticsraw_rgb_20201030.log) |
|  OmniSource  | 82.9  | 95.8  | [ckpt](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/omnisource/slowonly_r50_8x8x1_256e_minikinetics_omnisource_rgb_20201030-284cfd3b.pth) | [json](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/omnisource/slowonly_r50_8x8x1_256e_minikinetics_omnisource_rgb_20201030.json) | [log](https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/omnisource/slowonly_r50_8x8x1_256e_minikinetics_omnisource_rgb_20201030.log) |

We also list the benchmark in the original paper which run on Kinetics-400 for comparison:

|         Model          |  Baseline   |   +GG-img   | +[GG-IG]-img |   +IG-vid   |    +KRaw    | OmniSource  |
| :--------------------: | :---------: | :---------: | :----------: | :---------: | :---------: | :---------: |
|   TSN-3seg-ResNet50    | 70.6 / 89.4 | 71.5 / 89.5 | 72.0 / 90.0  | 72.0 / 90.3 | 71.7 / 89.6 | 73.6 / 91.0 |
| SlowOnly-4x16-ResNet50 | 73.8 / 90.9 | 74.5 / 91.4 | 75.2 / 91.6  | 75.2 / 91.7 | 74.5 / 91.1 | 76.6 / 92.5 |

## Citing OmniSource

If you find OmniSource useful for your research, please consider citing the paper using the following BibTeX entry.

[ALGORITHM]

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```
