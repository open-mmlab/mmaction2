# ACRN

[Actor-centric relation network](https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Sun_Actor-centric_Relation_Network_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Current state-of-the-art approaches for spatio-temporal action localization rely on detections at the frame level and model temporal context with 3D ConvNets. Here, we go one step further and model spatio-temporal relations to capture the interactions between human actors, relevant objects and scene elements essential to differentiate similar human actions. Our approach is weakly supervised and mines the relevant elements automatically with an actor-centric relational network (ACRN). ACRN computes and accumulates pair-wise relation information from actor and global scene features, and generates relation features for action classification. It is implemented as neural networks and can be trained jointly with an existing action detection system. We show that ACRN outperforms alternative approaches which capture relation information, and that the proposed framework improves upon the state-of-the-art performance on JHMDB and AVA. A visualization of the learned relation features confirms that our approach is able to attend to the relevant relations for each action.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142996406-09ac1b09-2a9e-478c-9035-5fe7a80bc80b.png" width="800"/>
</div>

## Results and Models

### AVA2.1

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|          8x8x1          |  8   | SlowFast ResNet50 | Kinetics-400 | 27.65 | [config](/configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb_20220906-0dae1a90.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.log) |

### AVA2.2

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|          8x8x1          |  8   | SlowFast ResNet50 | Kinetics-400 | 27.71 | [config](/configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb_20220906-66ec24a2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.

For more details on data preparation, you can refer to [AVA](/tools/data/ava/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ACRN with SlowFast backbone on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ACRN with SlowFast backbone on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{sun2018actor,
  title={Actor-centric relation network},
  author={Sun, Chen and Shrivastava, Abhinav and Vondrick, Carl and Murphy, Kevin and Sukthankar, Rahul and Schmid, Cordelia},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={318--334},
  year={2018}
}
```
