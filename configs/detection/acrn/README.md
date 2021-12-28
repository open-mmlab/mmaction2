# ACRN

## Abstract

<!-- [ABSTRACT] -->

Current state-of-the-art approaches for spatio-temporal action localization rely on detections at the frame level and model temporal context with 3D ConvNets. Here, we go one step further and model spatio-temporal relations to capture the interactions between human actors, relevant objects and scene elements essential to differentiate similar human actions. Our approach is weakly supervised and mines the relevant elements automatically with an actor-centric relational network (ACRN). ACRN computes and accumulates pair-wise relation information from actor and global scene features, and generates relation features for action classification. It is implemented as neural networks and can be trained jointly with an existing action detection system. We show that ACRN outperforms alternative approaches which capture relation information, and that the proposed framework improves upon the state-of-the-art performance on JHMDB and AVA. A visualization of the learned relation features confirms that our approach is able to attend to the relevant relations for each action.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142996406-09ac1b09-2a9e-478c-9035-5fe7a80bc80b.png" width="800"/>
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
@inproceedings{sun2018actor,
  title={Actor-centric relation network},
  author={Sun, Chen and Shrivastava, Abhinav and Vondrick, Carl and Murphy, Kevin and Sukthankar, Rahul and Schmid, Cordelia},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={318--334},
  year={2018}
}
```

## Model Zoo

### AVA2.1

|                            Model                             | Modality |  Pretrained  | Backbone | Input | gpus | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb](/configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 27.1 | [log](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava_rgb-49b07bf2.pth) |

### AVA2.2

|                            Model                             | Modality |  Pretrained  | Backbone | Input | gpus | mAP  |                             log                              |                             json                             |                             ckpt                             |
| :----------------------------------------------------------: | :------: | :----------: | :------: | :---: | :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb](/configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py) |   RGB    | Kinetics-400 | ResNet50 | 32x2  |  8   | 27.8 | [log](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.log) | [json](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb-2be32625.pth) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

:::

For more details on data preparation, you can refer to AVA in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ACRN with SlowFast backbone on AVA with periodic validation.

```shell
python tools/train.py configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py --validate
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ACRN with SlowFast backbone on AVA and dump the result to a csv file.

```shell
python tools/test.py configs/detection/acrn/slowfast_acrn_kinetics_pretrained_r50_8x8x1_cosine_10e_ava22_rgb.py checkpoints/SOME_CHECKPOINT.pth --eval mAP --out results.csv
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
