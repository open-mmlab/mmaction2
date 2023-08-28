# TSM

[TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The explosive growth in video streaming gives rise to challenges on performing video understanding at high accuracy and low computation cost. Conventional 2D CNNs are computationally cheap but cannot capture temporal relationships; 3D CNN based methods can achieve good performance but are computationally intensive, making it expensive to deploy. In this paper, we propose a generic and effective Temporal Shift Module (TSM) that enjoys both high efficiency and high performance. Specifically, it can achieve the performance of 3D CNN but maintain 2D CNN's complexity. TSM shifts part of the channels along the temporal dimension; thus facilitate information exchanged among neighboring frames. It can be inserted into 2D CNNs to achieve temporal modeling at zero computation and zero parameters. We also extended TSM to online setting, which enables real-time low-latency online video recognition and video object detection. TSM is accurate and efficient: it ranks the first place on the Something-Something leaderboard upon publication; on Jetson Nano and Galaxy Note8, it achieves a low latency of 13ms and 35ms for online video recognition.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019083-abc0de39-9ea1-4175-be5c-073c90de64c3.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus |           backbone            | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |            config            |                       ckpt |                        log |
| :---------------------: | :--------: | :--: | :---------------------------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :--------------------------: | -------------------------: | -------------------------: |
|          1x1x8          |  224x224   |  8   |           ResNet50            | ImageNet |  73.18   |  90.56   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   |           ResNet50            | ImageNet |  73.22   |  90.22   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb_20220831-a6db1e5d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.log) |
|         1x1x16          |  224x224   |  8   |           ResNet50            | ImageNet |  75.12   |  91.55   | 16 clips x 10 crop | 65.75G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb_20220831-042b1748.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb.log) |
|      1x1x8 (dense)      |  224x224   |  8   |           ResNet50            | ImageNet |  73.38   |  90.78   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb_20220831-f55d3c2b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet50 (NonLocalDotProduct) | ImageNet |  74.49   |  91.15   | 8 clips x 10 crop  | 61.30G | 31.68M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb_20220831-108bfde5.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   |   ResNet50 (NonLocalGauss)    | ImageNet |  73.66   |  90.99   | 8 clips x 10 crop  | 59.06G | 28.00M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb_20220831-7e54dacf.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  74.34   |  91.23   | 8 clips x 10 crop  | 61.30G | 31.68M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb_20220831-35eddb57.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   |          MobileNetV2          | ImageNet |  68.71   |  88.32   |  8 clips x 3 crop  | 3.269G | 2.736M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb_20230414-401127fd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.log) |
|         1x1x16          |  224x224   |  8   |         MobileOne-S4          | ImageNet |  74.38   |  91.71   | 16 clips x 10 crop | 48.65G | 13.72M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb_20230825-a7f8876b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-mobileone-s4_8xb16-1x1x16-50e_kinetics400-rgb.log) |

### Something-something V2

| frame sampling strategy | resolution | gpus | backbone  | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |               config                |               ckpt                |                log                |
| :---------------------: | :--------: | :--: | :-------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :---------------------------------: | :-------------------------------: | :-------------------------------: |
|          1x1x8          |  224x224   |  8   | ResNet50  | ImageNet |  62.72   |  87.70   | 8 clips x 3 crop  | 32.88G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20230317-be0fc26e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.log) |
|         1x1x16          |  224x224   |  8   | ResNet50  | ImageNet |  64.16   |  88.61   | 16 clips x 3 crop | 65.75G | 23.87M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20230317-ec6696ad.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet101 | ImageNet |  63.70   |  88.28   | 8 clips x 3 crop  | 62.66G | 42.86M | [config](/configs/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb_20230320-efcc0d1b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
3. MoibleOne backbone supports reparameterization during inference. You can use the provided [reparameterize tool](/tools/convert/reparameterize_model.py) to convert the checkpoint and switch to the [deploy config file](/configs/recognition/tsm/tsm_imagenet-pretrained-mobileone-s4_deploy_8xb16-1x1x16-50e_kinetics400-rgb.py).

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py \
     --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}
```

<!-- [BACKBONE] -->

```BibTeX
@article{Nonlocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```
