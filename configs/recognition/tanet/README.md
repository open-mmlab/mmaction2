# TANet

[TAM: Temporal Adaptive Module for Video Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_TAM_Temporal_Adaptive_Module_for_Video_Recognition_ICCV_2021_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video data is with complex temporal dynamics due to various factors such as camera motion, speed variation, and different activities. To effectively capture this diverse motion pattern, this paper presents a new temporal adaptive module ({\\bf TAM}) to generate video-specific temporal kernels based on its own feature map. TAM proposes a unique two-level adaptive modeling scheme by decoupling the dynamic kernel into a location sensitive importance map and a location invariant aggregation weight. The importance map is learned in a local temporal window to capture short-term information, while the aggregation weight is generated from a global view with a focus on long-term structure. TAM is a modular block and could be integrated into 2D CNNs to yield a powerful video architecture (TANet) with a very small extra computational cost. The extensive experiments on Kinetics-400 and Something-Something datasets demonstrate that our TAM outperforms other temporal modeling methods consistently, and achieves the state-of-the-art performance under the similar complexity.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018253-c3e1ba5b-ac35-4c55-be28-0134b76888e8.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |      reference top1 acc       |      reference top5 acc       | testing protocol | FLOPs | params |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------------------: | :---------------------------: | :--------------: | :---: | :----: | :---------------: | :-------------: | :------------: |
|       dense-1x1x8       |  224x224   |  8   | ResNet50 | ImageNet |  76.25   |  92.41   | [76.22](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | [92.53](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | 8 clips x 3 crop | 43.0G | 25.6M  | [config](/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb_20220919-a34346bc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.log) |

### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain |  top1 acc   |  top5 acc   | testing protocol  | FLOPs | params |               config               |               ckpt               |               log               |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------: | :---------: | :---------------: | :---: | :----: | :--------------------------------: | :------------------------------: | :-----------------------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet | 46.98/49.71 | 75.75/77.43 | 16 clips x 3 crop | 43.1G | 25.1M  | [config](/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb_20220906-de50e4ef.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb.log) |
|         1x1x16          |  224x224   |  8   | ResNet50 | ImageNet | 48.24/50.95 | 78.16/79.28 | 16 clips x 3 crop | 86.1G | 25.1M  | [config](/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb_20220919-cc37e9b8.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. The checkpoints for reference repo can be downloaded [here](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing).
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to

- [Kinetics400](/tools/data/kinetics/README.md)
- [Something-something V1](/tools/data/sthv1/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TANet model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TANet model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{liu2020tam,
  title={TAM: Temporal Adaptive Module for Video Recognition},
  author={Liu, Zhaoyang and Wang, Limin and Wu, Wayne and Qian, Chen and Lu, Tong},
  journal={arXiv preprint arXiv:2005.06803},
  year={2020}
}
```
