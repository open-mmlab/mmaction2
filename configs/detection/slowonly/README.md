# SlowOnly

[Slowfast networks for video recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>

## Results and Models

### AVA2.1

| frame sampling strategy | gpus |                backbone                |   pretrain   |  mAP  |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :--: | :------------------------------------: | :----------: | :---: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|         4x16x1          |  8   |           SlowOnly ResNet50            | Kinetics-400 | 20.72 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-953ef5fe.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   |           SlowOnly ResNet50            | Kinetics-700 | 22.77 | [config](/configs/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-b3b6d44e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 21.55 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb_20220906-5ae3f91b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 23.77 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb_20220906-9760eadb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   |           SlowOnly ResNet101           | Kinetics-400 | 24.83 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb_20220906-43f16877.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |

### AVA2.2 (Trained on AVA-Kinetics)

Currently, we only use the training set of AVA-Kinetics and evaluate on the AVA2.2 validation dataset. The AVA-Kinetics validation dataset will be supported soon.

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|         4x16x1          |  8   | SlowOnly ResNet50 | Kinetics-400 | 24.53 | [config](/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-33e3ca7c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|         4x16x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 25.87 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-a07e8c15.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-400 | 26.10 | [config](/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-8f8dff3b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |

### AVA2.2 (Trained on AVA-Kinetics with tricks)

We conduct ablation studies to show the improvements of training tricks using SlowOnly8x8 pretrained on the Kinetics700 dataset. The baseline is the last row in **AVA2.2 (Trained on AVA-Kinetics)**.

|         method         | frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                  config                  |                  ckpt                   |                  log                   |
| :--------------------: | :---------------------: | :--: | :---------------: | :----------: | :---: | :--------------------------------------: | :-------------------------------------: | :------------------------------------: |
|        baseline        |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|       + context        |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.31 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5d514f8c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
| + temporal max pooling |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.48 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5b5e71eb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|    + nonlinear head    |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 29.83 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-87624265.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|      + focal loss      |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 30.33 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb_20221205-37aa8395.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.log) |
|     + more frames      |         16x4x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 31.29 | [config](/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb_20221205-dd652f81.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.log) |

### MultiSports

| frame sampling strategy | gpus |     backbone      |   pretrain   | f-mAP | v-mAP@0.2 | v-mAP@0.5 | v-mAP@0.1:0.9 | gpu_mem(M) |               config               |               ckpt               |               log                |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :-------: | :-------: | :-----------: | :--------: | :--------------------------------: | :------------------------------: | :------------------------------: |
|         4x16x1          |  8   | SlowOnly ResNet50 | Kinetics-400 | 26.40 |   15.48   |   10.62   |     9.65      |    8509    | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-a1ca5e76.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. **+ context** indicates that using both RoI feature and global pooled feature for classification; **+ temporal max pooling** indicates that using max pooling in the temporal dimension for the feature; **nonlinear head** indicates that using a 2-layer mlp instead of a linear classifier.
3. MultiSports dataset utilizes frame-mAP(f-mAP) and video-mAP(v-map) to evaluate performance. Frame-mAP evaluates on detection results of each frame, and video-mAP uses 3D IoU to evaluate tube-level results under several thresholds. You could refer to the [competition page](https://codalab.lisn.upsaclay.fr/competitions/3736#learn_the_details-evaluation) for details.

For more details on data preparation, you can refer to

- [AVA](/tools/data/ava/README.md)
- [AVA-Kinetics](/tools/data/ava_kinetics/README.md)
- [MultiSports](/tools/data/multisports/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={ICCV},
  pages={6202--6211},
  year={2019}
}
```

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={CVPR},
  pages={6047--6056},
  year={2018}
}
```

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```
