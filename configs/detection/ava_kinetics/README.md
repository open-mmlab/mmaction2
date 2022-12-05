# AVA

[The AVA-Kinetics Localized Human Actions Video Dataset](https://arxiv.org/abs/2005.00214)

<!-- [ALGORITHM] -->

<div align="center">
  <img src="https://user-images.githubusercontent.com/35267818/205511687-8cafd48c-7f4a-4a4c-a8e6-8182635b0411.png" width="800px"/>
</div>

## Abstract

<!-- [ABSTRACT] -->

This paper describes the AVA-Kinetics localized human actions video dataset. The dataset is collected by annotating videos from the Kinetics-700 dataset using the AVA annotation protocol, and extending the original AVA dataset with these new AVA annotated Kinetics clips. The dataset contains over 230k clips annotated with the 80 AVA action classes for each of the humans in key-frames. We describe the annotation process and provide statistics about the new dataset. We also include a baseline evaluation using the Video Action Transformer Network on the AVA-Kinetics dataset, demonstrating improved performance for action classification on the AVA test set.

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```

## Results and Models

### AVA2.2

Currently, we only use the training set of AVA-Kinetics and evaluate on the AVA2.2 validation dataset. The AVA-Kinetics validation dataset will be supported soon.

| frame sampling strategy | resolution | gpus |     backbone      |   pretrain   |  mAP  |                    config                    |                    ckpt                     |                    log                     |
| :---------------------: | :--------: | :--: | :---------------: | :----------: | :---: | :------------------------------------------: | :-----------------------------------------: | :----------------------------------------: |
|         4x16x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-400 | 24.53 | [config](/configs/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-33e3ca7c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|         4x16x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 25.87 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-a07e8c15.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-400 | 26.10 | [config](/configs/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-8f8dff3b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |

### Training with tricks

We conduct ablation studies to show the improvements of training tricks using SlowOnly8x8 pretrained on the Kinetics700 dataset. The baseline is the last raw in [AVA2.2](https://github.com/hukkai/mmaction2/tree/ava-kinetics-exp/configs/detection/ava_kinetics#ava22).

|         method         | frame sampling strategy | resolution | gpus |     backbone      |   pretrain   |  mAP  |                config                 |                ckpt                 |                 log                 |
| :--------------------: | :---------------------: | :--------: | :--: | :---------------: | :----------: | :---: | :-----------------------------------: | :---------------------------------: | :---------------------------------: |
|        baseline        |          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|       + context        |          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.31 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5d514f8c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
| + temporal max pooling |          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.48 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5b5e71eb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|    + nonlinear head    |          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 29.83 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-87624265.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|      + focal loss      |          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 30.33 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb_20221205-37aa8395.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.log) |
|     + more frames      |         16x4x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 31.29 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb_20221205-dd652f81.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.log) |

Note:

The **gpus** indicates the number of gpu we used to get the checkpoint; **+ context** indicates that using both RoI feature and global pooled feature for classification; **+ temporal max pooling** indicates that using max pooling in the temporal dimension for the feature; **nonlinear head** indicates that using a 2-layer mlp instead of a linear classifier.

For more details on data preparation, you can refer to [AVA-Kinetics Data Preparation](/tools/data/ava_kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA-Kinetics in a deterministic option.

```shell
python tools/train.py configs/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA-Kinetics and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

<!-- [DATASET] -->

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```
