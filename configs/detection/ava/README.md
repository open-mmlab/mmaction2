# AVA

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

<!-- [ALGORITHM] -->

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```

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

## Model Zoo

### AVA2.1

|                            Model                             | Modality |  Pretrained  | Backbone  | Input | gpus |   Resolution   | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | short-side 256 | 20.1 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) |
| [slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet50  | 4x16  |  8   | short-side 256 | 21.8 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_omnisource_pretrained_r50_4x16x1_20e_ava_rgb_20201217-0c6d2e98.pth) |
| [slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 4x16  |  8   | short-side 256 | 21.75 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/20210316_122517.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/20210316_122517.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb/slowonly_nl_kinetics_pretrained_r50_4x16x1_10e_ava_rgb_20210316-959829ec.pth) |
| [slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb](/configs/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 8x8  |  8x2   | short-side 256 | 23.79 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/20210316_122517.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/20210316_122517.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb/slowonly_nl_kinetics_pretrained_r50_8x8x1_10e_ava_rgb_20210316-5742e4dd.pth) |
| [slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet101 |  8x8  | 8x2  | short-side 256 | 24.6 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_kinetics_pretrained_r101_8x8x1_20e_ava_rgb_20201217-1c9b4117.pth) |
| [slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb](/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py) |   RGB    |  OmniSource  | ResNet101 |  8x8  | 8x2  | short-side 256 | 25.9 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth) |
| [slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | short-side 256 | 24.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth) |
| [slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | short-side 256 | 25.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_context_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201222-f4d209c9.pth) |
| [slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50  | 32x2  | 8x2  | short-side 256 | 25.5 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb_20201217-ae225e97.pth) |

### AVA2.2

|                            Model                             | Modality |  Pretrained  | Backbone | Input | gpus | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 26.1 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-b987b516.pth) |
| [slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 26.4 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-874e0845.pth) |
| [slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 26.8 | [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_temporal_max_focal_alpha3_gamma1_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-345618cd.pth) |

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

Example: train SlowOnly model on AVA with periodic validation.

```shell
python tools/train.py configs/detection/ava/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py --validate
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting) .

### Train Custom Classes From Ava Dataset

You can train custom classes from ava. Ava suffers from class imbalance. There are more then 100,000 samples for classes like `stand`/`listen to (a person)`/`talk to (e.g., self, a person, a group)`/`watch (a person)`, whereas half of all classes has less than 500 samples. In most cases, training custom classes with fewer samples only will lead to better results.

Three steps to train custom classes:

- Step 1: Select custom classes from original classes, named `custom_classes`. Class `0` should not be selected since it is reserved for further usage (to identify whether a proposal is positive or negative, not implemented yet) and will be added automatically.
- Step 2: Set `num_classes`. In order to be compatible with current codes, Please make sure `num_classes == len(custom_classes) + 1`.
  - The new class `0` corresponds to original class `0`. The new class `i`(i > 0) corresponds to original class `custom_classes[i-1]`.
  - There are three `num_classes` in ava config, `model -> roi_head -> bbox_head -> num_classes`, `data -> train -> num_classes` and `data -> val -> num_classes`.
  - If `num_classes <= 5`, input arg `topk` of `BBoxHeadAVA` should be modified. The default value of `topk` is `(3, 5)`, and all elements of `topk` must be smaller than `num_classes`.
- Step 3: Make sure all custom classes are in `label_file`. It is worth mentioning that there are two label files, `ava_action_list_v2.1_for_activitynet_2018.pbtxt`(contains 60 classes, 20 classes are missing) and `ava_action_list_v2.1.pbtxt`(contains all 80 classes).

Take `slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb` as an example, training custom classes with AP in range `(0.1, 0.3)`, aka `[3, 6, 10, 27, 29, 38, 41, 48, 51, 53, 54, 59, 61, 64, 70, 72]`. Please note that, the previously mentioned AP is calculated by original ckpt, which is trained by all 80 classes. The results are listed as follows.

|training classes|mAP(custom classes)|config|log|json|ckpt|
|:-:|:-:|:-:|:-:|:-:|:-:|
|All 80 classes|0.1948|[slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py)|[log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201127.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) |
|custom classes|0.3311|[slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes-4ab80419.pth) |
|All 80 classes|0.1864|[slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-6e7c704d.pth) |
|custom classes|0.3785|[slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes](/configs/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes.py)| [log](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305.log) | [json](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/ava/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes/slowfast_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_custom_classes_20210305-c6225546.pth) |

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on AVA and dump the result to a csv file.

```shell
python tools/test.py configs/detection/ava/slowonly_kinetics_pretrained_r50_8x8x1_20e_ava_rgb.py checkpoints/SOME_CHECKPOINT.pth --eval mAP --out results.csv
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
