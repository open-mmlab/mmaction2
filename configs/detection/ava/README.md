# AVA

[Ava: A video dataset of spatio-temporally localized atomic visual actions](https://openaccess.thecvf.com/content_cvpr_2018/html/Gu_AVA_A_Video_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/>
</div>

## Abstract

<!-- [ABSTRACT] -->

This paper introduces a video dataset of spatio-temporally localized Atomic Visual Actions (AVA). The AVA dataset densely annotates 80 atomic visual actions in 430 15-minute video clips, where actions are localized in space and time, resulting in 1.58M action labels with multiple labels per person occurring frequently. The key characteristics of our dataset are: (1) the definition of atomic visual actions, rather than composite actions; (2) precise spatio-temporal annotations with possibly multiple annotations for each person; (3) exhaustive annotation of these atomic actions over 15-minute video clips; (4) people temporally linked across consecutive segments; and (5) using movies to gather a varied set of action representations. This departs from existing datasets for spatio-temporal action recognition, which typically provide sparse annotations for composite actions in short video clips. We will release the dataset publicly.
AVA, with its realistic scene and action complexity, exposes the intrinsic difficulty of action recognition. To benchmark this, we present a novel approach for action localization that builds upon the current state-of-the-art methods, and demonstrates better performance on JHMDB and UCF101-24 categories. While setting a new state of the art on existing datasets, the overall results on AVA are low at 15.6% mAP, underscoring the need for developing new approaches for video understanding.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143015933-36eb7abd-d38f-4be6-a327-4d34c6f4edc1.png" width="800"/>
</div>

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

## Results and Models

### AVA2.1

|                               Model                                | Modality |  Pretrained  | Backbone  | Input | gpus |   Resolution   |  mAP  |                  log                   |                  ckpt                   |
| :----------------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :---: | :------------------------------------: | :-------------------------------------: |
| [slowonly_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowonly_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | short-side 256 | 20.76 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowonly_nl_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | short-side 256 | 21.49 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowonly_nl_kinetics400_pretrained_r50_8x8x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics400_pretrained_r50_8x8x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  |  8x8  |  8   | short-side 256 | 23.74 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowonly_kinetics400_pretrained_r101_8x8x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowonly_kinetics400_pretrained_r101_8x8x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet101 |  8x8  |  8   | short-side 256 | 24.82 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowfast_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowfast_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  |  8   | short-side 256 | 24.27 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowfast_context_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb](/configs/detection/ava/slowfast_context_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  |  8   | short-side 256 | 25.25 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowfast_kinetics400_pretrained_r50_8x8x1_20e_8xb8_ava_rgb](/configs/detection/ava/slowfast_kinetics400_pretrained_r50_8x8x1_20e_8xb8_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  |  8   | short-side 256 | 25.73 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |

### AVA2.2

|                                       Model                                       | Modality |  Pretrained  | Backbone | Input | gpus |  mAP  |                  log                   |                  ckpt                   |
| :-------------------------------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :---: | :------------------------------------: | :-------------------------------------: |
| [slowfast_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb](/configs/detection/ava/slowfast_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 25.98 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 26.38 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |
| [slowfast_temporal_max_focal_alpha3_gamma1_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 26.59 | [log](https://download.openmmlab.com/) | [ckpt](https://download.openmmlab.com/) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. **Context** indicates that using both RoI feature and global pooled feature for classification, which leads to around 1% mAP improvement in general.

:::

For more details on data preparation, you can refer to AVA in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA in a deterministic option.

```shell
python tools/train.py configs/detection/ava/slowonly_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting) .

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA.

```shell
python tools/test.py configs/detection/ava/slowonly_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py checkpoints/SOME_CHECKPOINT.pth
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .

## Citation

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```
