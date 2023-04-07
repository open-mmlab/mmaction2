# C3D

[Learning Spatiotemporal Features with 3D Convolutional Networks](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We propose a simple, yet effective approach for spatiotemporal feature learning using deep 3-dimensional convolutional networks (3D ConvNets) trained on a large scale supervised video dataset. Our findings are three-fold: 1) 3D ConvNets are more suitable for spatiotemporal feature learning compared to 2D ConvNets; 2) A homogeneous architecture with small 3x3x3 convolution kernels in all layers is among the best performing architectures for 3D ConvNets; and 3) Our learned features, namely C3D (Convolutional 3D), with a simple linear classifier outperform state-of-the-art methods on 4 different benchmarks and are comparable with current best methods on the other 2 benchmarks. In addition, the features are compact: achieving 52.8% accuracy on UCF101 dataset with only 10 dimensions and also very efficient to compute due to the fast inference of ConvNets. Finally, they are conceptually very simple and easy to train and use.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043383-8c26f5d6-d45e-47ae-be18-c23456eb84b9.png" width="800"/>
</div>

## Results and Models

### UCF-101

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |                config                |                ckpt                |                log                |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :----------------------------------: | :--------------------------------: | :-------------------------------: |
|         16x1x1          |  112x112   |  8   |   c3d    | sports1m |  83.08   |  95.93   | 10 clips x 1 crop | 38.5G | 78.4M  | [config](/configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.log) |

1. The author of C3D normalized UCF-101 with volume mean and used SVM to classify videos, while we normalized the dataset with RGB mean value and used a linear classifier.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.

For more details on data preparation, you can refer to [UCF101](/tools/data/ucf101/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C3D model on UCF-101 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C3D model on UCF-101 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

<!-- [ALGORITHM] -->

```BibTeX
@ARTICLE{2014arXiv1412.0767T,
author = {Tran, Du and Bourdev, Lubomir and Fergus, Rob and Torresani, Lorenzo and Paluri, Manohar},
title = {Learning Spatiotemporal Features with 3D Convolutional Networks},
keywords = {Computer Science - Computer Vision and Pattern Recognition},
year = 2014,
month = dec,
eid = {arXiv:1412.0767}
}
```
