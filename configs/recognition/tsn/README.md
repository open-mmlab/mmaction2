# TSN

[Temporal segment networks: Towards good practices for deep action recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Deep convolutional networks have achieved great success for visual recognition in still images. However, for action recognition in videos, the advantage over traditional methods is not so evident. This paper aims to discover the principles to design effective ConvNet architectures for action recognition in videos and learn these models given limited training samples. Our first contribution is temporal segment network (TSN), a novel framework for video-based action recognition. which is based on the idea of long-range temporal structure modeling. It combines a sparse temporal sampling strategy and video-level supervision to enable efficient and effective learning using the whole action video. The other contribution is our study on a series of good practices in learning ConvNets on video data with the help of temporal segment network. Our approach obtains the state-the-of-art performance on the datasets of HMDB51 ( 69.4%) and UCF101 (94.2%). We also visualize the learned ConvNet models, which qualitatively demonstrates the effectiveness of temporal segment network and the proposed good practices.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019237-8823045b-dfa3-45cc-a992-ee83ab9d8459.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | scheduler | resolution | gpus | backbone  | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |              config              |                           ckpt |                           log |
| :---------------------: | :-------: | :--------: | :--: | :-------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :------------------------------: | -----------------------------: | ----------------------------: |
|          1x1x3          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  72.83   |  90.65   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x5          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  73.80   |  91.21   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb_20220906-65d68713.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  74.12   |  91.34   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.log) |
|       dense-1x1x5       | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  71.37   |  89.67   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb_20220906-dcbc6e01.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet101 | ImageNet |  75.89   |  92.07   | 25 clips x 10 crop | 195.8G | 43.32M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.log) |

### Something-Something V2

| frame sampling strategy | scheduler | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |              config              |                           ckpt |                            log |
| :---------------------: | :-------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :------------------------------: | -----------------------------: | -----------------------------: |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet50 | ImageNet |  35.51   |  67.09   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb_20230313-06ad7d03.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.log) |
|         1x1x16          | MultiStep |  224x224   |  8   | ResNet50 | ImageNet |  36.91   |  68.77   | 25 clips x 10 crop | 102.7G | 24.33M | [config](/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb_20230221-85bcc1c3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb.log) |

### Using backbones from 3rd-party in TSN

It's possible and convenient to use a 3rd-party backbone for TSN under the framework of MMAction2, here we provide some examples for:

- [x] Backbones from [MMClassification](https://github.com/open-mmlab/mmclassification/)
- [x] Backbones from [MMPretrain](https://github.com/open-mmlab/mmpretrain)
- [x] Backbones from [TorchVision](https://github.com/pytorch/vision/)
- [x] Backbones from [TIMM (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models)

| frame sampling strategy | scheduler | resolution | gpus |     backbone     | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |            config             |                         ckpt |                         log |
| :---------------------: | :-------: | :--------: | :--: | :--------------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :---------------------------: | ---------------------------: | --------------------------: |
|          1x1x3          | MultiStep |  224x224   |  8   |    ResNext101    | ImageNet |  72.95   |  90.36   | 25 clips x 10 crop | 200.3G | 42.95M | [config](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb_20221209-de2d5615.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x3          | MultiStep |  224x224   |  8   |   DenseNet161    | ImageNet |  72.07   |  90.15   | 25 clips x 10 crop | 194.6G | 27.36M | [config](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb_20220906-5f4c0daf.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x3          | MultiStep |  224x224   |  8   | Swin Transformer | ImageNet |  77.03   |  92.61   | 25 clips x 10 crop | 386.7G | 87.15M | [config](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb_20220906-65ed814e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   | Swin Transformer | ImageNet |  79.22   |  94.20   | 25 clips x 10 crop | 386.7G | 87.15M | [config](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb_20230530-428f0064.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   |   MobileOne-S4   | ImageNet |  73.65   |  91.32   | 25 clips x 10 crop |  76G   | 13.72M | [config](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb_20230825-2da3c1f7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-mobileone-s4_8xb32-1x1x8-100e_kinetics400-rgb.log) |

1. Note that some backbones in TIMM are not supported due to multiple reasons. Please refer to [PR #880](https://github.com/open-mmlab/mmaction2/pull/880) for details.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. MoibleOne backbone supports reparameterization during inference. You can use the provided [reparameterize tool](/tools/convert/reparameterize_model.py) to convert the checkpoint and switch to the [deploy config file](/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-mobileone-s4_deploy_8xb32-1x1x8-100e_kinetics400-rgb.py).

For more details on data preparation, you can refer to

- [Kinetics](/tools/data/kinetics/README.md)
- [Something-something V2](/tools/data/sthv2/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py  \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{wang2016temporal,
  title={Temporal segment networks: Towards good practices for deep action recognition},
  author={Wang, Limin and Xiong, Yuanjun and Wang, Zhe and Qiao, Yu and Lin, Dahua and Tang, Xiaoou and Van Gool, Luc},
  booktitle={European conference on computer vision},
  pages={20--36},
  year={2016},
  organization={Springer}
}
```
