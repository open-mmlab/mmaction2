# SlowFast

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

| frame sampling strategy | gpus |             backbone             |   pretrain   |  mAP  |                   config                    |                   ckpt                    |                   log                    |
| :---------------------: | :--: | :------------------------------: | :----------: | :---: | :-----------------------------------------: | :---------------------------------------: | :--------------------------------------: |
|         4x16x1          |  8   |        SlowFast ResNet50         | Kinetics-400 | 24.32 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-5180ea3c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   | SlowFast ResNet50 (with context) | Kinetics-400 | 25.34 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb_20220906-5bb4f6f2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   |        SlowFast ResNet50         | Kinetics-400 | 25.80 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.log) |

### AVA2.2

| frame sampling strategy | gpus |                 backbone                  |   pretrain   |  mAP  |                  config                  |                  ckpt                  |                  log                  |
| :---------------------: | :--: | :---------------------------------------: | :----------: | :---: | :--------------------------------------: | :------------------------------------: | :-----------------------------------: |
|          8x8x1          |  8   |             SlowFast ResNet50             | Kinetics-400 | 25.90 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |  8   |     SlowFast ResNet50 (temporal-max)      | Kinetics-400 | 26.41 | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-13a9078e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |  8   | SlowFast ResNet50 (temporal-max, focal loss) | Kinetics-400 | 26.65 | [config](/configs/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-dd59e26f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |

### MultiSports

| frame sampling strategy | gpus |     backbone      |   pretrain   | f-mAP | v-mAP@0.2 | v-mAP@0.5 | v-mAP@0.1:0.9 | gpu_mem(M) |               config               |               ckpt               |               log                |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :-------: | :-------: | :-----------: | :--------: | :--------------------------------: | :------------------------------: | :------------------------------: |
|         4x16x1          |  8   | SlowFast ResNet50 | Kinetics-400 | 36.88 |   22.83   |   16.9    |     14.74     |   18618    | [config](/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-af666368.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. **with context** indicates that using both RoI feature and global pooled feature for classification; **temporal-max** indicates that using max pooling in the temporal dimension for the feature.
3. MultiSports dataset utilizes frame-mAP(f-mAP) and video-mAP(v-map) to evaluate performance. Frame-mAP evaluates on detection results of each frame, and video-mAP uses 3D IoU to evaluate tube-level results under several thresholds. You could refer to the [competition page](https://codalab.lisn.upsaclay.fr/competitions/3736#learn_the_details-evaluation) for details.

For more details on data preparation, you can refer to

- [AVA](/tools/data/ava/README.md)
- [MultiSports](/tools/data/multisports/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowFast model on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowFast model on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
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
