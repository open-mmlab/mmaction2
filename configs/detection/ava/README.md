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

| frame sampling strategy | resolution | gpus |               backbone               |   pretrain   |  mAP  | gpu_mem(M) |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :----------------------------------: | :----------: | :---: | :--------: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         4x16x1          |    raw     |  8   |          SlowOnly ResNet50           | Kinetics-400 | 20.76 |    8503    | [config](/configs/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-953ef5fe.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   |          SlowOnly ResNet50           | Kinetics-700 | 22.77 |    8503    | [config](/configs/detection/ava/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-b3b6d44e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 21.49 |   11870    | [config](/configs/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb_20220906-5ae3f91b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 23.74 |   25375    | [config](/configs/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb_20220906-9760eadb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.log) |
|          8x8x1          |    raw     |  8   |          SlowOnly ResNet101          | Kinetics-400 | 24.82 |   23477    | [config](/configs/detection/ava/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb_20220906-43f16877.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   |          SlowFast ResNet50           | Kinetics-400 | 24.27 |   18616    | [config](/configs/detection/ava/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-5180ea3c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   |   SlowFast ResNet50 (with context)   | Kinetics-400 | 25.25 |   18616    | [config](/configs/detection/ava/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb_20220906-5bb4f6f2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |    raw     |  8   |          SlowFast ResNet50           | Kinetics-400 | 25.73 |   13802    | [config](/configs/detection/ava/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.log) |

### AVA2.2

| frame sampling strategy | resolution | gpus |               backbone               |   pretrain   |  mAP  | gpu_mem(M) |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :----------------------------------: | :----------: | :---: | :--------: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|          8x8x1          |    raw     |  8   |          SlowFast ResNet50           | Kinetics-400 | 25.82 |   10484    | [config](/configs/detection/ava/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |    raw     |  8   |   SlowFast ResNet50 (temporal-max)   | Kinetics-400 | 26.32 |   10484    | [config](/configs/detection/ava/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-13a9078e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowFast ResNet50 (temporal-max, focal loss) | Kinetics-400 | 26.58 |   10484    | [config](/configs/detection/ava/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-dd59e26f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |

Note:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. **With context** indicates that using both RoI feature and global pooled feature for classification, which leads to around 1% mAP improvement in general.

:::

For more details on data preparation, you can refer to [AVA Data Preparation](/tools/data/ava/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA in a deterministic option.

```shell
python tools/train.py configs/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

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
