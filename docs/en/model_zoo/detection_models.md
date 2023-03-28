# Spatio Temporal Action Detection Models

## ACRN

[Actor-centric relation network](https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Sun_Actor-centric_Relation_Network_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Current state-of-the-art approaches for spatio-temporal action localization rely on detections at the frame level and model temporal context with 3D ConvNets. Here, we go one step further and model spatio-temporal relations to capture the interactions between human actors, relevant objects and scene elements essential to differentiate similar human actions. Our approach is weakly supervised and mines the relevant elements automatically with an actor-centric relational network (ACRN). ACRN computes and accumulates pair-wise relation information from actor and global scene features, and generates relation features for action classification. It is implemented as neural networks and can be trained jointly with an existing action detection system. We show that ACRN outperforms alternative approaches which capture relation information, and that the proposed framework improves upon the state-of-the-art performance on JHMDB and AVA. A visualization of the learned relation features confirms that our approach is able to attend to the relevant relations for each action.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142996406-09ac1b09-2a9e-478c-9035-5fe7a80bc80b.png" width="800"/>
</div>

### Results and Models

#### AVA2.1

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|          8x8x1          |  8   | SlowFast ResNet50 | Kinetics-400 | 27.65 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb_20220906-0dae1a90.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.log) |

#### AVA2.2

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|          8x8x1          |  8   | SlowFast ResNet50 | Kinetics-400 | 27.71 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb_20220906-66ec24a2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava22-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.

For more details on data preparation, you can refer to [AVA](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ava/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ACRN with SlowFast backbone on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ACRN with SlowFast backbone on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/acrn/slowfast-acrn_kinetics400-pretrained-r50_8xb8-8x8x1-cosine-10e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{sun2018actor,
  title={Actor-centric relation network},
  author={Sun, Chen and Shrivastava, Abhinav and Vondrick, Carl and Murphy, Kevin and Sukthankar, Rahul and Schmid, Cordelia},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={318--334},
  year={2018}
}
```

## LFB

[Long-term feature banks for detailed video understanding](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

To understand the world, we humans constantly need to relate the present to the past, and put events in context. In this paper, we enable existing video models to do the same. We propose a long-term feature bank---supportive information extracted over the entire span of a video---to augment state-of-the-art video models that otherwise would only view short clips of 2-5 seconds. Our experiments demonstrate that augmenting 3D convolutional networks with a long-term feature bank yields state-of-the-art results on three challenging video datasets: AVA, EPIC-Kitchens, and Charades.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016220-21d90fb3-fd9f-499c-820f-f6c421bda7aa.png" width="800"/>
</div>

### Results and Models

#### AVA2.1

| frame sampling strategy | resolution | gpus |               backbone               |   pretrain   |  mAP  | gpu_mem(M) |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :----------------------------------: | :----------: | :---: | :--------: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         4x16x1          |    raw     |  8   | SlowOnly ResNet50 (with Nonlocal LFB) | Kinetics-400 | 24.05 |    8620    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb_20220906-4c5b9f25.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   |   SlowOnly ResNet50 (with Max LFB)   | Kinetics-400 | 22.15 |    8425    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb_20220906-4963135b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.log) |

Note:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. We use `slowonly_r50_4x16x1` instead of `I3D-R50-NL` in the original paper as the backbone of LFB, but we have achieved the similar improvement: (ours: 20.1 -> 24.05 vs. author: 22.1 -> 25.8).
3. Because the long-term features are randomly sampled in testing, the test accuracy may have some differences.
4. Before train or test lfb, you need to infer feature bank with the [slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/lfb/slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py). For more details on infer feature bank, you can refer to [Train](##Train) part.
5. You can also dowonload long-term feature bank from [AVA_train_val_float32_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float32_lfb.rar) or [AVA_train_val_float16_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float16_lfb.rar), and then put them on `lfb_prefix_path`.
6. The ROIHead now supports single-label classification (i.e. the network outputs at most
   one-label per actor). This can be done by (a) setting multilabel=False during training and
   the test_cfg.rcnn.action_thr for testing.

### Train

#### a. Infer long-term feature bank for training

Before train or test lfb, you need to infer long-term feature bank first.

Specifically, run the test on the training, validation, testing dataset with the config file [slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/lfb/slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py) (The config file will only infer the feature bank of training dataset and you need set `dataset_mode = 'val'` to infer the feature bank of validation dataset in the config file.), and the shared head [LFBInferHead](https://github.com/open-mmlab/mmaction2/tree/master/mmaction/models/roi_heads/shared_heads/lfb_infer_head.py) will generate the feature bank.

A long-term feature bank file of AVA training and validation datasets with float32 precision occupies 3.3 GB. If store the features with float16 precision, the feature bank occupies 1.65 GB.

You can use the following command to infer feature bank of AVA training and validation dataset and the feature bank will be stored in `lfb_prefix_path/lfb_train.pkl` and `lfb_prefix_path/lfb_val.pkl`.

```shell
## set `dataset_mode = 'train'` in lfb_slowonly_r50_ava_infer.py
python tools/test.py slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth --eval mAP

## set `dataset_mode = 'val'` in lfb_slowonly_r50_ava_infer.py
python tools/test.py slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth --eval mAP
```

We use [slowonly_r50_4x16x1 checkpoint](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) from [slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/ava/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) to infer feature bank.

#### b. Train LFB

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train LFB model on AVA with half-precision long-term feature bank.

```shell
python tools/train.py configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py \
  --validate --seed 0 --deterministic
```

For more details and optional arguments infos, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

#### a. Infer long-term feature bank for testing

Before train or test lfb, you also need to infer long-term feature bank first. If you have generated the feature bank file, you can skip it.

The step is the same with **Infer long-term feature bank for training** part in [Train](##Train).

#### b. Test LFB

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test LFB model on AVA with half-precision long-term feature bank and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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
@inproceedings{wu2019long,
  title={Long-term feature banks for detailed video understanding},
  author={Wu, Chao-Yuan and Feichtenhofer, Christoph and Fan, Haoqi and He, Kaiming and Krahenbuhl, Philipp and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={284--293},
  year={2019}
}
```

## SlowFast

[Slowfast networks for video recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>

### Results and Models

#### AVA2.1

| frame sampling strategy | gpus |             backbone             |   pretrain   |  mAP  |                   config                    |                   ckpt                    |                   log                    |
| :---------------------: | :--: | :------------------------------: | :----------: | :---: | :-----------------------------------------: | :---------------------------------------: | :--------------------------------------: |
|         4x16x1          |  8   |        SlowFast ResNet50         | Kinetics-400 | 24.32 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-5180ea3c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   | SlowFast ResNet50 (with context) | Kinetics-400 | 25.34 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb_20220906-5bb4f6f2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50-context_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   |        SlowFast ResNet50         | Kinetics-400 | 25.80 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb_20220906-39133ec7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb/slowfast_kinetics400-pretrained-r50_8xb8-8x8x1-20e_ava21-rgb.log) |

#### AVA2.2

| frame sampling strategy | gpus |                 backbone                  |   pretrain   |  mAP  |                  config                  |                  ckpt                  |                  log                  |
| :---------------------: | :--: | :---------------------------------------: | :----------: | :---: | :--------------------------------------: | :------------------------------------: | :-----------------------------------: |
|          8x8x1          |  8   |             SlowFast ResNet50             | Kinetics-400 | 25.90 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-d934a48f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |  8   |     SlowFast ResNet50 (temporal-max)      | Kinetics-400 | 26.41 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-13a9078e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_kinetics400-pretrained-r50-temporal-max_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |
|          8x8x1          |  8   | SlowFast ResNet50 (temporal-max, focal loss) | Kinetics-400 | 26.65 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb_20220906-dd59e26f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowfast/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb/slowfast_r50-k400-pre-temporal-max-focal-alpha3-gamma1_8xb6-8x8x1-cosine-10e_ava22-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. **with context** indicates that using both RoI feature and global pooled feature for classification; **temporal-max** indicates that using max pooling in the temporal dimension for the feature.

For more details on data preparation, you can refer to [AVA](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ava/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowFast model on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowFast model on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/slowfast/slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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

## SlowOnly

[Slowfast networks for video recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>

### Results and Models

#### AVA2.1

| frame sampling strategy | gpus |                backbone                |   pretrain   |  mAP  |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :--: | :------------------------------------: | :----------: | :---: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|         4x16x1          |  8   |           SlowOnly ResNet50            | Kinetics-400 | 20.72 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-953ef5fe.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   |           SlowOnly ResNet50            | Kinetics-700 | 22.77 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb_20220906-b3b6d44e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics700-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 21.55 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb_20220906-5ae3f91b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-4x16x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 (NonLocalEmbedGauss) | Kinetics-400 | 23.77 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb_20220906-9760eadb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r50-nl_8xb16-8x8x1-20e_ava21-rgb.log) |
|          8x8x1          |  8   |           SlowOnly ResNet101           | Kinetics-400 | 24.83 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb_20220906-43f16877.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.log) |

#### AVA2.2 (Trained on AVA-Kinetics)

Currently, we only use the training set of AVA-Kinetics and evaluate on the AVA2.2 validation dataset. The AVA-Kinetics validation dataset will be supported soon.

| frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                      config                      |                      ckpt                      |                      log                      |
| :---------------------: | :--: | :---------------: | :----------: | :---: | :----------------------------------------------: | :--------------------------------------------: | :-------------------------------------------: |
|         4x16x1          |  8   | SlowOnly ResNet50 | Kinetics-400 | 24.53 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-33e3ca7c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|         4x16x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 25.87 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb_20221205-a07e8c15.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-4x16x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-400 | 26.10 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-8f8dff3b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k400-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |

#### AVA2.2 (Trained on AVA-Kinetics with tricks)

We conduct ablation studies to show the improvements of training tricks using SlowOnly8x8 pretrained on the Kinetics700 dataset. The baseline is the last row in **AVA2.2 (Trained on AVA-Kinetics)**.

|         method         | frame sampling strategy | gpus |     backbone      |   pretrain   |  mAP  |                  config                  |                  ckpt                   |                  log                   |
| :--------------------: | :---------------------: | :--: | :---------------: | :----------: | :---: | :--------------------------------------: | :-------------------------------------: | :------------------------------------: |
|        baseline        |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 27.82 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-16a01c37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|       + context        |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.31 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5d514f8c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
| + temporal max pooling |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 28.48 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-5b5e71eb.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|    + nonlinear head    |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 29.83 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb_20221205-87624265.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-10e_ava-kinetics-rgb.log) |
|      + focal loss      |          8x8x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 30.33 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb_20221205-37aa8395.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb/slowonly_k700-pre-r50-context-temporal-max-nl-head_8xb8-8x8x1-focal-10e_ava-kinetics-rgb.log) |
|     + more frames      |         16x4x1          |  8   | SlowOnly ResNet50 | Kinetics-700 | 31.29 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb_20221205-dd652f81.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb/slowonly_k700-pre-r50_8xb8-16x4x1-10e-tricks_ava-kinetics-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. **+ context** indicates that using both RoI feature and global pooled feature for classification; **+ temporal max pooling** indicates that using max pooling in the temporal dimension for the feature; **nonlinear head** indicates that using a 2-layer mlp instead of a linear classifier.

For more details on data preparation, you can refer to

- [AVA](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ava/README.md)
- [AVA-Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ava_kinetics/README.md)

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train the SlowOnly model on AVA2.1 in a deterministic option with periodic validation.

```shell
python tools/train.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test the SlowOnly model on AVA2.1 and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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
