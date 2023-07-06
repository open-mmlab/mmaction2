# MViT V2

[MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](http://openaccess.thecvf.com//content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video classification, as well as object detection. We present an improved version of MViT that incorporates decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 boxAP on COCO object detection as well as 86.1% on Kinetics-400 video classification.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/33249023/196627033-03a4e9b1-082e-42ee-a2a0-77f874fe632a.png" width="50%"/>
</div>

## Results and Models

1. Models with * in `Inference results` are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data, and models in `Training results` are trained in MMAction2 on our data.
2. The values in columns named after `reference` are copied from paper, and `reference*` are results using [SlowFast](https://github.com/facebookresearch/SlowFast/) repo and trained on our data.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. MaskFeat fine-tuning experiment is based on pretrain model from [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/main/projects/maskfeat_video), and the corresponding reference result is based on pretrain model from [SlowFast](https://github.com/facebookresearch/SlowFast/).
5. Due to the different versions of Kinetics-400, our training results are different from paper.
6. Due to the training efficiency, we currently only provide MViT-small training results, we don't ensure other config files' training accuracy and welcome you to contribute your reproduction results.
7. We use `repeat augment` in MViT training configs following [SlowFast](https://github.com/facebookresearch/SlowFast/). [Repeat augment](https://arxiv.org/pdf/1901.09335.pdf) takes multiple times of data augment for one video, this way can improve the generalization of the model and relieve the IO stress of loading videos. And please note that the actual batch size is `num_repeats` times of `batch_size` in `train_dataloader`.

### Inference results

#### Kinetics-400

| frame sampling strategy | resolution |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top5 acc        | testing protocol | FLOPs | params |        config        |        ckpt        |
| :---------------------: | :--------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :---: | :----: | :------------------: | :----------------: |
|         16x4x1          |  224x224   | MViTv2-S\* | From scratch |   81.1   |   94.7   | [81.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop |  64G  | 34.5M  | [config](/configs/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth) |
|         32x3x1          |  224x224   | MViTv2-B\* | From scratch |   82.6   |   95.8   | [82.9](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [95.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop | 225G  | 51.2M  | [config](/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-base-p244_32x3x1_kinetics400-rgb_20221021-f392cd2d.pth) |
|         40x3x1          |  312x312   | MViTv2-L\* | From scratch |   85.4   |   96.2   | [86.1](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [97.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 3 crop | 2828G |  213M  | [config](/configs/recognition/mvit/mvit-large-p244_40x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-large-p244_40x3x1_kinetics400-rgb_20221021-11fe1f97.pth) |

#### Something-Something V2

| frame sampling strategy | resolution |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top5 acc        | testing protocol | FLOPs | params |        config        |        ckpt        |
| :---------------------: | :--------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :---: | :----: | :------------------: | :----------------: |
|       uniform 16        |  224x224   | MViTv2-S\* |     K400     |   68.1   |   91.0   | [68.2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [91.4](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop |  64G  | 34.4M  | [config](/configs/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_u16_sthv2-rgb_20221021-65ecae7d.pth) |
|       uniform 32        |  224x224   | MViTv2-B\* |     K400     |   70.8   |   92.7   | [70.5](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [92.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | 225G  | 51.1M  | [config](/configs/recognition/mvit/mvit-base-p244_u32_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-base-p244_u32_sthv2-rgb_20221021-d5de5da6.pth) |
|       uniform 40        |  312x312   | MViTv2-L\* | IN21K + K400 |   73.2   |   94.0   | [73.3](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | 2828G |  213M  | [config](/configs/recognition/mvit/mvit-large-p244_u40_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-large-p244_u40_sthv2-rgb_20221021-61696e07.pth) |

### Training results

#### Kinetics-400

| frame sampling strategy | resolution | backbone |   pretrain    | top1 acc | top5 acc |     reference\* top1 acc      |      reference\* top5 acc      | testing protocol  | FLOPs | params |      config      |      ckpt      |      log      |
| :---------------------: | :--------: | :------: | :-----------: | :------: | :------: | :---------------------------: | :----------------------------: | :---------------: | :---: | :----: | :--------------: | :------------: | :-----------: |
|         16x4x1          |  224x224   | MViTv2-S | From scratch  |   80.6   |   94.7   | [80.8](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop  |  64G  | 34.5M  | [config](/configs/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb_20230201-23284ff3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.log) |
|         16x4x1          |  224x224   | MViTv2-S | K400 MaskFeat |   81.8   |   95.2   | [81.5](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | [94.9](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | 10 clips x 1 crop |  71G  | 36.4M  | [config](/configs/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb_20230201-5bced1d0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.log) |

the corresponding result without repeat augment is as follows:

| frame sampling strategy | resolution | backbone |   pretrain   | top1 acc | top5 acc |                 reference\* top1 acc                 |                 reference\* top5 acc                 | testing protocol | FLOPs | params |
| :---------------------: | :--------: | :------: | :----------: | :------: | :------: | :--------------------------------------------------: | :--------------------------------------------------: | :--------------: | :---: | :----: |
|         16x4x1          |  224x224   | MViTv2-S | From scratch |   79.4   |   93.9   | [80.8](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop |  64G  | 34.5M  |

#### Something-Something V2

| frame sampling strategy | resolution | backbone | pretrain | top1 acc | top5 acc |      reference top1 acc       |       reference top5 acc       | testing protocol | FLOPs | params |       config       |       ckpt       |       log       |
| :---------------------: | :--------: | :------: | :------: | :------: | :------: | :---------------------------: | :----------------------------: | :--------------: | :---: | :----: | :----------------: | :--------------: | :-------------: |
|       uniform 16        |  224x224   | MViTv2-S |   K400   |   68.2   |   91.3   | [68.2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [91.4](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop |  64G  | 34.4M  | [config](/configs/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb_20230201-4065c1b9.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb.log) |

For more details on data preparation, you can refer to

- [Kinetics](/tools/data/kinetics/README.md)
- [Something-something V2](/tools/data/sthv2/README.md)

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test MViT model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/mvit/mvit-small-p244_16x4x1_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```bibtex
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```
