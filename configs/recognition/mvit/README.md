# MViT V2

> [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](http://openaccess.thecvf.com//content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In this paper, we study Multiscale Vision Transformers (MViTv2) as a unified architecture for image and video
classification, as well as object detection. We present an improved version of MViT that incorporates
decomposed relative positional embeddings and residual pooling connections. We instantiate this architecture
in five sizes and evaluate it for ImageNet classification, COCO detection and Kinetics video recognition where
it outperforms prior work. We further compare MViTv2s' pooling attention to window attention mechanisms where
it outperforms the latter in accuracy/compute. Without bells-and-whistles, MViTv2 has state-of-the-art
performance in 3 domains: 88.8% accuracy on ImageNet classification, 58.7 boxAP on COCO object detection as
well as 86.1% on Kinetics-400 video classification.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/33249023/196627033-03a4e9b1-082e-42ee-a2a0-77f874fe632a.png" width="50%"/>
</div>

## Results and models

### Kinetics-400

| frame sampling strategy |   resolution   |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top1 acc        | testing protocol | params |        config        |        ckpt         |
| :---------------------: | :------------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :----: | :------------------: | :-----------------: |
|         16x4x1          | short-side 320 | MViTv2-S\* | From scratch |   81.1   |   94.7   | [81.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-small-p244_16x4x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |
|         32x3x1          | short-side 320 | MViTv2-B\* | From scratch |   82.6   |   95.8   | [82.9](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [95.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |
|         40x3x1          | short-side 320 | MViTv2-L\* | From scratch |   85.4   |   96.2   | [86.1](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [97.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 3 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-large-p244_40x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |

### Something-Something V2

| frame sampling strategy |   resolution   |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top1 acc        | testing protocol | params |        config        |        ckpt         |
| :---------------------: | :------------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :----: | :------------------: | :-----------------: |
|       uniform 16        | short-side 320 | MViTv2-S\* |     K400     |   68.1   |   91.0   | [68.2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [91.4](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-small-p244_16x4x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |
|       uniform 32        | short-side 320 | MViTv2-B\* |     K400     |   70.8   |   92.7   | [70.5](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [92.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |
|       uniform 40        | short-side 320 | MViTv2-L\* | IN21K + K400 |   73.2   |   94.0   | [73.3](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | xx.xM  | [config](/configs/recognition/mvit/mvit-large-p244_40x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/) |

*Models with * are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3D models, training will be available soon.*

1. The values in columns named after "reference" are copied from paper
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test MViT model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/mvit/mvit-small_16x4x1_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

```bibtex
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```
