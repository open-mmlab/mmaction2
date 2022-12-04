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
|          4x16x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 25.87 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-a07e8c15.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-400 | 26.10 | [config](/configs/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-8f8dff3b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |    raw     |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/ava_kinetics/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |

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
