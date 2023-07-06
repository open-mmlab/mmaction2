# LFB

[Long-term feature banks for detailed video understanding](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Long-Term_Feature_Banks_for_Detailed_Video_Understanding_CVPR_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

To understand the world, we humans constantly need to relate the present to the past, and put events in context. In this paper, we enable existing video models to do the same. We propose a long-term feature bank---supportive information extracted over the entire span of a video---to augment state-of-the-art video models that otherwise would only view short clips of 2-5 seconds. Our experiments demonstrate that augmenting 3D convolutional networks with a long-term feature bank yields state-of-the-art results on three challenging video datasets: AVA, EPIC-Kitchens, and Charades.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016220-21d90fb3-fd9f-499c-820f-f6c421bda7aa.png" width="800"/>
</div>

## Results and Models

### AVA2.1

| frame sampling strategy | resolution | gpus |               backbone               |   pretrain   |  mAP  | gpu_mem(M) |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :----------------------------------: | :----------: | :---: | :--------: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         4x16x1          |    raw     |  8   | SlowOnly ResNet50 (with Nonlocal LFB) | Kinetics-400 | 24.11 |    8620    | [config](/configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb_20220906-4c5b9f25.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.log) |
|         4x16x1          |    raw     |  8   |   SlowOnly ResNet50 (with Max LFB)   | Kinetics-400 | 22.15 |    8425    | [config](/configs/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb_20220906-4963135b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/detection/lfb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb/slowonly-lfb-max_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.log) |

Note:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. We use `slowonly_r50_4x16x1` instead of `I3D-R50-NL` in the original paper as the backbone of LFB, but we have achieved the similar improvement: (ours: 20.1 -> 24.05 vs. author: 22.1 -> 25.8).
3. Because the long-term features are randomly sampled in testing, the test accuracy may have some differences.
4. Before train or test lfb, you need to infer feature bank with the [slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py](/configs/detection/lfb/slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py). For more details on infer feature bank, you can refer to [Train](#Train) part.
5. The ROIHead now supports single-label classification (i.e. the network outputs at most
   one-label per actor). This can be done by (a) setting multilabel=False during training and
   the test_cfg.rcnn.action_thr for testing.

## Train

### a. Infer long-term feature bank for training

Before train or test lfb, you need to infer long-term feature bank first. You can also dowonload long-term feature bank from [AVA_train_val_float32_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float32_lfb.rar) or [AVA_train_val_float16_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float16_lfb.rar), and then put them on `lfb_prefix_path`. In this case, you can skip this step.

Specifically, run the test on the training, validation, testing dataset with the config file [slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py](/configs/detection/lfb/slowonly-lfb_ava-pretrained-r50_infer-4x16x1_ava21-rgb.py) (The config file will only infer the feature bank of training dataset and you need set `dataset_mode = 'val'` to infer the feature bank of validation dataset in the config file.), and the shared head [LFBInferHead](/mmaction/models/roi_heads/shared_heads/lfb_infer_head.py) will generate the feature bank.

A long-term feature bank file of AVA training and validation datasets with float32 precision occupies 3.3 GB. If store the features with float16 precision, the feature bank occupies 1.65 GB.

You can use the following command to infer feature bank of AVA training and validation dataset and the feature bank will be stored in `lfb_prefix_path/lfb_train.pkl` and `lfb_prefix_path/lfb_val.pkl`.

```shell
# set `dataset_mode = 'train'` in lfb_slowonly_r50_ava_infer.py
python tools/test.py configs/detection/lfb/slowonly-lfb-infer_r50_ava21-rgb.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth

# set `dataset_mode = 'val'` in lfb_slowonly_r50_ava_infer.py
python tools/test.py configs/detection/lfb/slowonly-lfb-infer_r50_ava21-rgb.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth
```

We use [slowonly_r50_4x16x1 checkpoint](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth) from [slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb](/configs/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py) to infer feature bank.

### b. Train LFB

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train LFB model on AVA with half-precision long-term feature bank.

```shell
python tools/train.py configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py \
  --seed 0 --deterministic
```

For more details and optional arguments infos, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

### a. Infer long-term feature bank for testing

Before train or test lfb, you also need to infer long-term feature bank first. If you have generated the feature bank file, you can skip it.

The step is the same with **Infer long-term feature bank for training** part in [Train](#Train).

### b. Test LFB

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test LFB model on AVA with half-precision long-term feature bank and dump the result to a pkl file.

```shell
python tools/test.py configs/detection/lfb/slowonly-lfb-nl_kinetics400-pretrained-r50_8xb12-4x16x1-20e_ava21-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

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
@inproceedings{wu2019long,
  title={Long-term feature banks for detailed video understanding},
  author={Wu, Chao-Yuan and Feichtenhofer, Christoph and Fan, Haoqi and He, Kaiming and Krahenbuhl, Philipp and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={284--293},
  year={2019}
}
```
