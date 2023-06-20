# TPN

[Temporal Pyramid Network for Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Temporal_Pyramid_Network_for_Action_Recognition_CVPR_2020_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Visual tempo characterizes the dynamics and the temporal scale of an action. Modeling such visual tempos of different actions facilitates their recognition. Previous works often capture the visual tempo through sampling raw videos at multiple rates and constructing an input-level frame pyramid, which usually requires a costly multi-branch network to handle. In this work we propose a generic Temporal Pyramid Network (TPN) at the feature-level, which can be flexibly integrated into 2D or 3D backbone networks in a plug-and-play manner. Two essential components of TPN, the source of features and the fusion of features, form a feature hierarchy for the backbone so that it can capture action instances at various tempos. TPN also shows consistent improvements over other challenging baselines on several action recognition datasets. Specifically, when equipped with TPN, the 3D ResNet-50 with dense sampling obtains a 2% gain on the validation set of Kinetics-400. A further analysis also reveals that TPN gains most of its improvements on action classes that have large variances in their visual tempos, validating the effectiveness of TPN.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018779-1d2a398f-dbd3-405a-87e5-e188b61fcc86.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |   reference top1 acc    |   reference top5 acc    | testing protocol  | inference time(video/s) | gpu_mem(M) |    config    |    ckpt    |    log    |
| :---------------------: | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :---------------------: | :---------------: | :---------------------: | :--------: | :----------: | :--------: | :-------: |
|          8x8x1          | short-side 320 | 8x2  | ResNet50 |   None   |  74.20   |  91.48   |            x            |            x            | 10 clips x 3 crop |            x            |    6916    | [config](/configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.log) |
|          8x8x1          | short-side 320 |  8   | ResNet50 | ImageNet |  76.74   |  92.57   | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | 10 clips x 3 crop |            x            |    6916    | [config](/configs/recognition/tpn/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-fed3f4c1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.log) |

### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 100 | 8x6  | ResNet50 |   TSM    |  51.87   |  79.67   |         x          |         x          | 8 clips x 3 crop |            x            |    8828    | [config](/configs/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb_20230221-940a3615.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [Kinetics](/tools/data/kinetics/README.md)
- [Something-something V1](/tools/data/sthv1/README.md)
- [Something-something V2](/tools/data/sthv2/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TPN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    --work-dir work_dirs/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb [--validate --seed 0 --deterministic]
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TPN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```
