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

| frame sampling strategy |  gpus |               backbone               |   pretrain   |  mAP  |               config                |               ckpt                |               log                |
| :---------------------: |  :--: | :----------------------------------: | :----------: | :---: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         4x16x1          |  8   |          SlowOnly ResNet50           | Kinetics-400 | 20.72 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-953ef5fe.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    8   |          SlowOnly ResNet50           | Kinetics-700 | 22.77 | [config](/configs/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-b3b6d44e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 21.55 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb_20220906-5ae3f91b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |    8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 23.77 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb_20220906-9760eadb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.log) |
|          8x8x1          |    8   |          SlowOnly ResNet101          | Kinetics-400 | 24.83 | [config](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb_20220906-43f16877.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.

For more details on data preparation, you can refer to [AVA](/tools/data/ava/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).
## Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```
