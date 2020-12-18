# AVA

## Introduction

```
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}

@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}

@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## Model Zoo

### AVA2.1

|                            Model                             | Modality |  Pretrained  | Backbone  | Input | gpus |   Resolution   | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | short-side 256 | 20.1 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) |
| [slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet50  | 4x16  |  8   | short-side 256 | 21.8 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201217-0c6d2e98.pth) |
| [slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet101 |  8x8  | 8x2  | short-side 256 | 24.6 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201217-1c9b4117.pth) |
| [slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet101 |  8x8  | 8x2  | short-side 256 | 25.9 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth) |
| [slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | short-side 256 | 24.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth) |
| [slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | short-side 256 | 25.5 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth) |

- Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

For more details on data preparation, you can refer to AVA in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowOnly model on AVA with periodic validation.

```shell
python tools/train.py configs/detection/AVA/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py --validate
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting) .

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on AVA and dump the result to a csv file.

```shell
python tools/test.py configs/detection/AVA/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py checkpoints/SOME_CHECKPOINT.pth --eval bbox --out results.csv
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
