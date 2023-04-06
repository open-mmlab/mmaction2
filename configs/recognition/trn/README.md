# TRN

[Temporal Relational Reasoning in Videos](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Temporal relational reasoning, the ability to link meaningful transformations of objects or entities over time, is a fundamental property of intelligent species. In this paper, we introduce an effective and interpretable network module, the Temporal Relation Network (TRN), designed to learn and reason about temporal dependencies between video frames at multiple time scales. We evaluate TRN-equipped networks on activity recognition tasks using three recent video datasets - Something-Something, Jester, and Charades - which fundamentally depend on temporal relational reasoning. Our results demonstrate that the proposed TRN gives convolutional neural networks a remarkable capacity to discover temporal relations in videos. Through only sparsely sampled video frames, TRN-equipped networks can accurately predict human-object interactions in the Something-Something dataset and identify various human gestures on the Jester dataset with very competitive performance. TRN-equipped networks also outperform two-stream networks and 3D convolution networks in recognizing daily activities in the Charades dataset. Further analyses show that the models learn intuitive and interpretable visual common sense knowledge in videos.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018998-d2120c3d-a9a7-4e4c-90b1-1e5ff1fd5f06.png" width="800"/>
</div>

## Results and Models

### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) |  testing protocol  | FLOPs  | params |        config         |        ckpt         |         log         |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :----------------: | :----: | :----: | :-------------------: | :-----------------: | :-----------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet |         31.60 / 33.65         |         60.15 / 62.22         | 16 clips x 10 crop | 42.94G | 26.64M | [config](/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb_20220815-e13db2e9.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.log) |

### Something-Something V2

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) |  testing protocol  | FLOPs  | params |        config         |        ckpt         |         log         |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :----------------: | :----: | :----: | :-------------------: | :-----------------: | :-----------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet |         47.65 / 51.20         |         76.27 / 78.42         | 16 clips x 10 crop | 42.94G | 26.64M | [config](/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20220815-e01617db.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop only) and accurate setting (three crop and `twice_sample`).
3. In the original [repository](https://github.com/zhoubolei/TRN-pytorch), the author augments data with random flipping on something-something dataset, but the augmentation method may be wrong due to the direct actions, such as `push left to right`. So, we replaced `flip` with `flip with label mapping`, and change the testing method `TenCrop`, which has five flipped crops, to `Twice Sample & ThreeCrop`.
4. We use `ResNet50` instead of `BNInception` as the backbone of TRN. When Training `TRN-ResNet50` on sthv1 dataset in the original repository, we get top1 (top5) accuracy 30.542 (58.627) vs. ours 31.81 (60.47).

For more details on data preparation, you can refer to [Something-something V1](/tools/data/sthv1/README.md) and [Something-something V2](/tools/data/sthv2/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TRN model on sthv1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TRN model on sthv1 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```
