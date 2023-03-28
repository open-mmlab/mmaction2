# Action Recognition Models

## C2D

<!-- [ALGORITHM] -->

[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

### Abstract

<!-- [ABSTRACT] -->

Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our non-local models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/37479394/195281946-b70c76fd-d424-4371-95cf-09f51f20eda0.jpg" width="800"/>

</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | scheduler | resolution | gpus |    backbone     | pretrain | top1 acc | top5 acc |    reference top1 acc     |    reference top5 acc     | testing protocol  | FLOPs | params |     config     |     ckpt     |     log     |
| :---------------------: | :-------: | :--------: | :--: | :-------------: | :------: | :------: | :------: | :-----------------------: | :-----------------------: | :---------------: | :---: | :----: | :------------: | :----------: | :---------: |
|          8x8x1          | MultiStep |  224x224   |  8   |  ResNet50<br>   | ImageNet |  73.44   |  91.00   | 67.2<br>[\[PySlowFast\]](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md##kinetics-400-and-600) | 87.8<br>[\[PySlowFast\]](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | 10 clips x 3 crop |  33G  | 24.3M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb_20221027-e0227b22.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|          8x8x1          | MultiStep |  224x224   |  8   |  ResNet101<br>  | ImageNet |  74.97   |  91.77   |             x             |             x             | 10 clips x 3 crop |  63G  | 43.3M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb_20221027-557bd8bc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|          8x8x1          | MultiStep |  224x224   |  8   | ResNet50<br>(TemporalPool) | ImageNet |  73.89   |  91.21   | 71.9<br>[\[Non-Local\]](https://github.com/facebookresearch/video-nonlocal-net##modifications-for-improving-speed) | 90.0<br>[\[Non-Local\]](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | 10 clips x 3 crop |  19G  | 24.3M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb_20221027-3ca304fa.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|         16x4x1          | MultiStep |  224x224   |  8   | ResNet50<br>(TemporalPool) | ImageNet |  74.97   |  91.91   |             x             |             x             | 10 clips x 3 crop |  39G  | 24.3M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb_20221027-5f382a43.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb.log) |

1. The values in columns named after "reference" are the results reported in the original repo, using the same model settings.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C2D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py  \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C2D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{XiaolongWang2017NonlocalNN,
  title={Non-local Neural Networks},
  author={Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2017}
}
```

## C3D

[Learning Spatiotemporal Features with 3D Convolutional Networks](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We propose a simple, yet effective approach for spatiotemporal feature learning using deep 3-dimensional convolutional networks (3D ConvNets) trained on a large scale supervised video dataset. Our findings are three-fold: 1) 3D ConvNets are more suitable for spatiotemporal feature learning compared to 2D ConvNets; 2) A homogeneous architecture with small 3x3x3 convolution kernels in all layers is among the best performing architectures for 3D ConvNets; and 3) Our learned features, namely C3D (Convolutional 3D), with a simple linear classifier outperform state-of-the-art methods on 4 different benchmarks and are comparable with current best methods on the other 2 benchmarks. In addition, the features are compact: achieving 52.8% accuracy on UCF101 dataset with only 10 dimensions and also very efficient to compute due to the fast inference of ConvNets. Finally, they are conceptually very simple and easy to train and use.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043383-8c26f5d6-d45e-47ae-be18-c23456eb84b9.png" width="800"/>
</div>

### Results and Models

#### UCF-101

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |                config                |                ckpt                |                log                |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :----------------------------------: | :--------------------------------: | :-------------------------------: |
|         16x1x1          |  112x112   |  8   |   c3d    | sports1m |  83.08   |  95.93   | 10 clips x 1 crop | 38.5G | 78.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb_20220811-31723200.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.log) |

1. The author of C3D normalized UCF-101 with volume mean and used SVM to classify videos, while we normalized the dataset with RGB mean value and used a linear classifier.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.

For more details on data preparation, you can refer to [UCF101](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ucf101/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C3D model on UCF-101 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/c3d/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C3D model on UCF-101 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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

## CSN

[Video Classification With Channel-Separated Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Video_Classification_With_Channel-Separated_Convolutional_Networks_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Group convolution has been shown to offer great computational savings in various 2D convolutional architectures for image classification. It is natural to ask: 1) if group convolution can help to alleviate the high computational cost of video classification networks; 2) what factors matter the most in 3D group convolutional networks; and 3) what are good computation/accuracy trade-offs with 3D group convolutional networks. This paper studies the effects of different design choices in 3D group convolutional networks for video classification. We empirically demonstrate that the amount of channel interactions plays an important role in the accuracy of 3D group convolutional networks. Our experiments suggest two main findings. First, it is a good practice to factorize 3D convolutions by separating channel interactions and spatiotemporal interactions as this leads to improved accuracy and lower computational cost. Second, 3D channel-separated convolutions provide a form of regularization, yielding lower training accuracy but higher test accuracy compared to 3D convolutions. These two empirical findings lead us to design an architecture -- Channel-Separated Convolutional Network (CSN) -- which is simple, efficient, yet accurate. On Sports1M, Kinetics, and Something-Something, our CSNs are comparable with or better than the state-of-the-art while being 2-3 times more efficient.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143017317-1bd7e557-7d99-4964-8b89-ab5280945d54.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus |        backbone         | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |             config             |             ckpt              |             log              |
| :---------------------: | :--------: | :--: | :---------------------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :----------------------------: | :---------------------------: | :--------------------------: |
|         32x2x1          |  224x224   |  8   |     ResNet152 (IR)      |  IG65M   |  82.87   |  95.90   | 10 clips x 3 crop | 97.63G | 29.70M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb_20220811-c7a3cc5b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet152 (IR+BNFrozen) |  IG65M   |  82.84   |  95.92   | 10 clips x 3 crop | 97.63G | 29.70M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet50 (IR+BNFrozen)  |  IG65M   |  79.44   |  94.26   | 10 clips x 3 crop | 55.90G | 13.13M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  x   |     ResNet152 (IP)      |   None   |  77.80   |  93.10   | 10 clips x 3 crop | 109.9G | 33.02M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_r152_32x2x1-180e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-d565828d.pth) |              x               |
|         32x2x1          |  224x224   |  x   |     ResNet152 (IR)      |   None   |  76.53   |  92.28   | 10 clips x 3 crop | 97.6G  | 29.70M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_r152_32x2x1-180e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-5c933ae1.pth) |              x               |
|         32x2x1          |  224x224   |  x   | ResNet152 (IP+BNFrozen) |  IG65M   |  82.68   |  95.69   | 10 clips x 3 crop | 109.9G | 33.02M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth) |              x               |
|         32x2x1          |  224x224   |  x   | ResNet152 (IP+BNFrozen) | Sports1M |  79.07   |  93.82   | 10 clips x 3 crop | 109.9G | 33.02M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth) |              x               |
|         32x2x1          |  224x224   |  x   | ResNet152 (IR+BNFrozen) | Sports1M |  78.57   |  93.44   | 10 clips x 3 crop | 109.9G | 33.02M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-b9b10241.pth) |              x               |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
3. The **infer_ckpt** means those checkpoints are ported from [VMZ](https://github.com/facebookresearch/VMZ).

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CSN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py  \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CSN model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{inproceedings,
author = {Wang, Heng and Feiszli, Matt and Torresani, Lorenzo},
year = {2019},
month = {10},
pages = {5551-5560},
title = {Video Classification With Channel-Separated Convolutional Networks},
doi = {10.1109/ICCV.2019.00565}
}
```

<!-- [OTHERS] -->

```BibTeX
@inproceedings{ghadiyaram2019large,
  title={Large-scale weakly-supervised pre-training for video action recognition},
  author={Ghadiyaram, Deepti and Tran, Du and Mahajan, Dhruv},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12046--12055},
  year={2019}
}
```

## I3D

[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)

[Non-local Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

The paucity of videos in current action classification datasets (UCF-101 and HMDB-51) has made it difficult to identify good video architectures, as most methods obtain similar performance on existing small-scale benchmarks. This paper re-evaluates state-of-the-art architectures in light of the new Kinetics Human Action Video dataset. Kinetics has two orders of magnitude more data, with 400 human action classes and over 400 clips per class, and is collected from realistic, challenging YouTube videos. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics. We also introduce a new Two-Stream Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.9% on HMDB-51 and 98.0% on UCF-101.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043624-1944704a-5d3e-4a3f-b258-1505c49f6092.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus |           backbone            | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |            config            |            ckpt             |            log             |
| :---------------------: | :--------: | :--: | :---------------------------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :--------------------------: | :-------------------------: | :------------------------: |
|         32x2x1          |  224x224   |  8   | ResNet50 (NonLocalDotProduct) | ImageNet |  74.80   |  92.07   | 10 clips x 3 crop | 59.3G  | 35.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  74.73   |  91.80   | 10 clips x 3 crop | 59.3G  | 35.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb_20220812-afd8f562.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |   ResNet50 (NonLocalGauss)    | ImageNet |  73.97   |  91.33   | 10 clips x 3 crop |  56.5  | 31.7M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb_20220812-0c5cbf5a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |           ResNet50            | ImageNet |  73.47   |  91.27   | 10 clips x 3 crop | 43.5G  | 28.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|      dense-32x2x1       |  224x224   |  8   |           ResNet50            | ImageNet |  73.77   |  91.35   | 10 clips x 3 crop | 43.5G  | 28.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb_20220812-9f46003f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |       ResNet50 (Heavy)        | ImageNet |  76.21   |  92.48   | 10 clips x 3 crop | 166.3G | 33.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb_20220812-ed501b31.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train I3D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

<!-- [BACKBONE] -->

```BibTeX
@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```

## MViT V2

> [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](http://openaccess.thecvf.com//content/CVPR2022/papers/Li_MViTv2_Improved_Multiscale_Vision_Transformers_for_Classification_and_Detection_CVPR_2022_paper.pdf)

<!-- [ALGORITHM] -->

### Abstract

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

### Results and Models

1. Models with * in `Inference results` are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data, and models in `Training results` are trained in MMAction2 on our data.
2. The values in columns named after `reference` are copied from paper, and `reference*` are results using [SlowFast](https://github.com/facebookresearch/SlowFast/) repo and trained on our data.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. MaskFeat fine-tuning experiment is based on pretrain model from [MMSelfSup](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/projects/maskfeat_video), and the corresponding reference result is based on pretrain model from [SlowFast](https://github.com/facebookresearch/SlowFast/).
5. Due to the different versions of Kinetics-400, our training results are different from paper.
6. Due to the training efficiency, we currently only provide MViT-small training results, we don't ensure other config files' training accuracy and welcome you to contribute your reproduction results.
7. We use `repeat augment` in MViT training configs following [SlowFast](https://github.com/facebookresearch/SlowFast/). [Repeat augment](https://arxiv.org/pdf/1901.09335.pdf) takes multiple times of data augment for one video, this way can improve the generalization of the model and relieve the IO stress of loading videos. And please note that the actual batch size is `num_repeats` times of `batch_size` in `train_dataloader`.

#### Inference results

##### Kinetics-400

| frame sampling strategy | resolution |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top5 acc        | testing protocol | FLOPs | params |        config        |        ckpt        |
| :---------------------: | :--------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :---: | :----: | :------------------: | :----------------: |
|         16x4x1          |  224x224   | MViTv2-S\* | From scratch |   81.1   |   94.7   | [81.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop |  64G  | 34.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_16x4x1_kinetics400-rgb_20221021-9ebaaeed.pth) |
|         32x3x1          |  224x224   | MViTv2-B\* | From scratch |   82.6   |   95.8   | [82.9](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [95.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop | 225G  | 51.2M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-base-p244_32x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-base-p244_32x3x1_kinetics400-rgb_20221021-f392cd2d.pth) |
|         40x3x1          |  312x312   | MViTv2-L\* | From scratch |   85.4   |   96.2   | [86.1](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [97.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 3 crop | 2828G |  213M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-large-p244_40x3x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-large-p244_40x3x1_kinetics400-rgb_20221021-11fe1f97.pth) |

##### Something-Something V2

| frame sampling strategy | resolution |  backbone  |   pretrain   | top1 acc | top5 acc |        reference top1 acc        |        reference top5 acc        | testing protocol | FLOPs | params |        config        |        ckpt        |
| :---------------------: | :--------: | :--------: | :----------: | :------: | :------: | :------------------------------: | :------------------------------: | :--------------: | :---: | :----: | :------------------: | :----------------: |
|       uniform 16        |  224x224   | MViTv2-S\* |     K400     |   68.1   |   91.0   | [68.2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [91.4](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop |  64G  | 34.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-small-p244_u16_sthv2-rgb_20221021-65ecae7d.pth) |
|       uniform 32        |  224x224   | MViTv2-B\* |     K400     |   70.8   |   92.7   | [70.5](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [92.7](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | 225G  | 51.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-base-p244_u32_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-base-p244_u32_sthv2-rgb_20221021-d5de5da6.pth) |
|       uniform 40        |  312x312   | MViTv2-L\* | IN21K + K400 |   73.2   |   94.0   | [73.3](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.0](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop | 2828G |  213M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-large-p244_u40_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/converted/mvit-large-p244_u40_sthv2-rgb_20221021-61696e07.pth) |

#### Training results

##### Kinetics-400

| frame sampling strategy | resolution | backbone |   pretrain    | top1 acc | top5 acc |     reference\* top1 acc      |      reference\* top5 acc      | testing protocol  | FLOPs | params |      config      |      ckpt      |      log      |
| :---------------------: | :--------: | :------: | :-----------: | :------: | :------: | :---------------------------: | :----------------------------: | :---------------: | :---: | :----: | :--------------: | :------------: | :-----------: |
|         16x4x1          |  224x224   | MViTv2-S | From scratch  |   80.6   |   94.7   | [80.8](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop  |  64G  | 34.5M  | [config](configs/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb_20230201-23284ff3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb/mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb.log) |
|         16x4x1          |  224x224   | MViTv2-S | K400 MaskFeat |   81.8   |   95.2   | [81.5](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | [94.9](https://github.com/facebookresearch/SlowFast/blob/main/projects/maskfeat/README.md) | 10 clips x 1 crop |  71G  | 36.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb_20230201-5bced1d0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb/mvit-small-p244_k400-maskfeat-pre_8xb32-16x4x1-100e_kinetics400-rgb.log) |

the corresponding result without repeat augment is as follows:

| frame sampling strategy | resolution | backbone |   pretrain   | top1 acc | top5 acc |                 reference\* top1 acc                 |                 reference\* top5 acc                 | testing protocol | FLOPs | params |
| :---------------------: | :--------: | :------: | :----------: | :------: | :------: | :--------------------------------------------------: | :--------------------------------------------------: | :--------------: | :---: | :----: |
|         16x4x1          |  224x224   | MViTv2-S | From scratch |   79.4   |   93.9   | [80.8](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [94.6](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 5 clips x 1 crop |  64G  | 34.5M  |

##### Something-Something V2

| frame sampling strategy | resolution | backbone | pretrain | top1 acc | top5 acc |      reference top1 acc       |       reference top5 acc       | testing protocol | FLOPs | params |       config       |       ckpt       |       log       |
| :---------------------: | :--------: | :------: | :------: | :------: | :------: | :---------------------------: | :----------------------------: | :--------------: | :---: | :----: | :----------------: | :--------------: | :-------------: |
|       uniform 16        |  224x224   | MViTv2-S |   K400   |   68.2   |   91.3   | [68.2](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | [91.4](https://github.com/facebookresearch/SlowFast/blob/main/projects/mvitv2/README.md) | 1 clips x 3 crop |  64G  | 34.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb_20230201-4065c1b9.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/mvit/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb/mvit-small-p244_k400-pre_16xb16-u16-100e_sthv2-rgb.log) |

For more details on data preparation, you can refer to

- [Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md)
- [Something-something V2](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv2/README.md)

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test MViT model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/mvit/mvit-small-p244_16x4x1_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```bibtex
@inproceedings{li2021improved,
  title={MViTv2: Improved multiscale vision transformers for classification and detection},
  author={Li, Yanghao and Wu, Chao-Yuan and Fan, Haoqi and Mangalam, Karttikeya and Xiong, Bo and Malik, Jitendra and Feichtenhofer, Christoph},
  booktitle={CVPR},
  year={2022}
}
```

## Omnisource

<!-- TODO: add links to the tech report -->

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We propose to train a recognizer that can classify images and videos. The recognizer is jointly trained on image and video datasets. Compared with pre-training on the same image dataset, this method can significantly improve the video recognition performance.

<!-- [IMAGE]

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>
-->

### Results and Models

#### Kinetics-400

| frame sampling strategy |   scheduler   | resolution | gpus | backbone | joint-training | top1 acc | top5 acc | testing protocol  | FLOPs  | params |            config             |            ckpt             |             log             |
| :---------------------: | :-----------: | :--------: | :--: | :------: | :------------: | :------: | :------: | :---------------: | :----: | :----: | :---------------------------: | :-------------------------: | :-------------------------: |
|          8x8x1          | Linear+Cosine |  224x224   |  8   | ResNet50 |    ImageNet    |  77.30   |  93.23   | 10 clips x 3 crop | 54.75G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb_20230208-61c4be0d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowOnly model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py \
    --seed=0 --deterministic
```

We found that the training of this Omnisource model could crash for unknown reasons. If this happens, you can resume training by adding the `--cfg-options resume=True` to the training script.

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/omnisource/slowonly_r50_8xb16-8x8x1-256e_imagenet-kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## R2plus1D

[A closer look at spatiotemporal convolutions for action recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

In this paper we discuss several forms of spatiotemporal convolutions for video analysis and study their effects on action recognition. Our motivation stems from the observation that 2D CNNs applied to individual frames of the video have remained solid performers in action recognition. In this work we empirically demonstrate the accuracy advantages of 3D CNNs over 2D CNNs within the framework of residual learning. Furthermore, we show that factorizing the 3D convolutional filters into separate spatial and temporal components yields significantly advantages in accuracy. Our empirical study leads to the design of a new spatiotemporal convolutional block "R(2+1)D" which gives rise to CNNs that achieve results comparable or superior to the state-of-the-art on Sports-1M, Kinetics, UCF101 and HMDB51.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043885-3d00413c-b556-445e-9673-f5805c08c195.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |                config                |                ckpt                |                log                |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :----------------------------------: | :--------------------------------: | :-------------------------------: |
|          8x8x1          |  224x224   |  8   | ResNet34 |   None   |  69.76   |  88.41   | 10 clips x 3 crop | 53.1G | 63.8M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb_20220812-47cfe041.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet34 |   None   |  75.46   |  92.28   | 10 clips x 3 crop | 213G  | 63.8M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb_20220812-4270588c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{tran2018closer,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={6450--6459},
  year={2018}
}
```

## SlowFast

[SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy |    scheduler     | resolution | gpus |       backbone       | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |           config           |           ckpt            |           log            |
| :---------------------: | :--------------: | :--------: | :--: | :------------------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :------------------------: | :-----------------------: | :----------------------: |
|         4x16x1          |  Linear+Cosine   |  224x224   |  8   |       ResNet50       |   None   |  75.55   |  92.35   | 10 clips x 3 crop | 36.3G | 34.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |       ResNet50       |   None   |  76.80   |  92.99   | 10 clips x 3 crop | 66.1G | 34.6M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|          8x8x1          | Linear+MultiStep |  224x224   |  8   |       ResNet50       |   None   |  76.65   |  92.86   | 10 clips x 3 crop | 66.1G | 34.6M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb_20220818-b62a501f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |      ResNet101       |   None   |  78.65   |  93.88   | 10 clips x 3 crop | 126G  | 62.9M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|         4x16x1          |  Linear+Cosine   |  224x224   |  32  | ResNet101 + ResNet50 |   None   |  77.03   |  92.99   | 10 clips x 3 crop | 64.9G | 62.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb_20220901-a77ac3ee.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
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

#### Kinetics-400

| frame sampling strategy |    scheduler     | resolution | gpus |          backbone          | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |          config          |          ckpt          |          log           |
| :---------------------: | :--------------: | :--------: | :--: | :------------------------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :----------------------: | :--------------------: | :--------------------: |
|         4x16x1          |  Linear+Cosine   |  224x224   |  8   |          ResNet50          |   None   |  72.97   |  90.88   | 10 clips x 3 crop | 27.38G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb/slowonly_r50_4x16x1_256e_8xb16_kinetics400-rgb_20220901-f6a40d08.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb/slowonly_r50_4x16x1_256e_8xb16_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |          ResNet50          |   None   |  75.15   |  92.11   | 10 clips x 3 crop | 54.75G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb_20220901-2132fc87.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb/slowonly_r50_8xb16-8x8x1-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |         ResNet101          |   None   |  76.59   |  92.80   | 10 clips x 3 crop |  112G  | 60.36M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb_20220901-e6281431.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb/slowonly_r101_8xb16-8x8x1-196e_kinetics400-rgb.log) |
|         4x16x1          | Linear+MultiStep |  224x224   |  8   |          ResNet50          | ImageNet |  75.12   |  91.72   | 10 clips x 3 crop | 27.38G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb_20220901-e7b65fad.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb/slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb.log) |
|          8x8x1          | Linear+MultiStep |  224x224   |  8   |          ResNet50          | ImageNet |  76.45   |  92.55   | 10 clips x 3 crop | 54.75G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb/slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb_20220901-df42dc84.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb/slowonly_imagenet-pretrained-r50_8xb16-8x8x1-steplr-150e_kinetics400-rgb.log) |
|         4x16x1          | Linear+MultiStep |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  75.07   |  91.69   | 10 clips x 3 crop | 43.23G | 39.81M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb_20220901-cf739c75.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-4x16x1-steplr-150e_kinetics400-rgb.log) |
|          8x8x1          | Linear+MultiStep |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  76.65   |  92.47   | 10 clips x 3 crop | 96.66G | 39.81M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb_20220901-df42dc84.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb/slowonly_r50-in1k-pre-nl-embedded-gaussian_8xb16-8x8x1-steplr-150e_kinetics400-rgb.log) |

#### Kinetics-700

| frame sampling strategy |    scheduler     | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |             config             |             ckpt             |             log              |
| :---------------------: | :--------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :----------------------------: | :--------------------------: | :--------------------------: |
|         4x16x1          | Linear+MultiStep |  224x224   | 8x2  | ResNet50 | ImageNet |  65.52   |  86.39   | 10 clips x 3 crop | 27.38G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb_20221013-98b1b0a7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb.log) |
|          8x8x1          | Linear+MultiStep |  224x224   | 8x2  | ResNet50 | ImageNet |  67.67   |  87.80   | 10 clips x 3 crop | 54.75G | 32.45M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb_20221013-15b93b10.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowOnly model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/slowonly/slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```

## VideoSwin

[Video Swin Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

The vision community is witnessing a modeling shift from CNNs to Transformers, where pure Transformer architectures have attained top accuracy on the major video recognition benchmarks. These video models are all built on Transformer layers that globally connect patches across the spatial and temporal dimensions. In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including on action recognition (84.9 top-1 accuracy on Kinetics-400 and 85.9 top-1 accuracy on Kinetics-600 with ~20xless pre-training data and ~3xsmaller model size) and temporal modeling (69.6 top-1 accuracy on Something-Something v2).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/191190475-3aecf940-c254-47fa-96a7-df2d2b3bae68.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone |   pretrain   | top1 acc | top5 acc |      reference top1 acc      |      reference top1 acc      | testing protocol | FLOPs | params |      config      |      ckpt      |      log       |
| :---------------------: | :--------: | :--: | :------: | :----------: | :------: | :------: | :--------------------------: | :--------------------------: | :--------------: | :---: | :----: | :--------------: | :------------: | :------------: |
|         32x2x1          |  224x224   |  8   |  Swin-T  | ImageNet-1k  |  78.90   |  93.77   | 78.84 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 93.76 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop |  88G  | 28.2M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-S  | ImageNet-1k  |  80.54   |  94.46   | 80.58 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.45 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop | 166G  | 49.8M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-B  | ImageNet-1k  |  80.57   |  94.49   | 80.55 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.66 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop | 282G  | 88.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-L  | ImageNet-22k |  83.46   |  95.91   |            83.1\*            |            95.9\*            | 4 clips x 3 crop | 604G  |  197M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-78ad8b11.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |

#### Kinetics-700

| frame sampling strategy | resolution | gpus | backbone |   pretrain   | top1 acc | top5 acc | testing protocol | FLOPs | params |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :------: | :----------: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         32x2x1          |  224x224   |  16  |  Swin-L  | ImageNet-22k |  75.92   |  92.72   | 4 clips x 3 crop | 604G  |  197M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.py.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. `*` means that the numbers are copied from the paper.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. Pre-trained image models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer##main-results-on-imagenet-with-pretrained-models).

For more details on data preparation, you can refer to [Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VideoSwin model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test VideoSwin model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{liu2022video,
  title={Video swin transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3202--3211},
  year={2022}
}
```

## TANet

[TAM: Temporal Adaptive Module for Video Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_TAM_Temporal_Adaptive_Module_for_Video_Recognition_ICCV_2021_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Video data is with complex temporal dynamics due to various factors such as camera motion, speed variation, and different activities. To effectively capture this diverse motion pattern, this paper presents a new temporal adaptive module ({\\bf TAM}) to generate video-specific temporal kernels based on its own feature map. TAM proposes a unique two-level adaptive modeling scheme by decoupling the dynamic kernel into a location sensitive importance map and a location invariant aggregation weight. The importance map is learned in a local temporal window to capture short-term information, while the aggregation weight is generated from a global view with a focus on long-term structure. TAM is a modular block and could be integrated into 2D CNNs to yield a powerful video architecture (TANet) with a very small extra computational cost. The extensive experiments on Kinetics-400 and Something-Something datasets demonstrate that our TAM outperforms other temporal modeling methods consistently, and achieves the state-of-the-art performance under the similar complexity.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018253-c3e1ba5b-ac35-4c55-be28-0134b76888e8.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |      reference top1 acc       |      reference top5 acc       | testing protocol | FLOPs | params |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------------------: | :---------------------------: | :--------------: | :---: | :----: | :---------------: | :-------------: | :------------: |
|       dense-1x1x8       |  224x224   |  8   | ResNet50 | ImageNet |  76.25   |  92.41   | [76.22](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | [92.53](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | 8 clips x 3 crop | 43.0G | 25.6M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb_20220919-a34346bc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.log) |

#### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain |  top1 acc   |  top5 acc   | testing protocol  | FLOPs | params |               config               |               ckpt               |               log               |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------: | :---------: | :---------------: | :---: | :----: | :--------------------------------: | :------------------------------: | :-----------------------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet | 46.98/49.71 | 75.75/77.43 | 16 clips x 3 crop | 43.1G | 25.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb_20220906-de50e4ef.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb8-1x1x8-50e_sthv1-rgb.log) |
|         1x1x16          |  224x224   |  8   | ResNet50 | ImageNet | 48.24/50.95 | 78.16/79.28 | 16 clips x 3 crop | 86.1G | 25.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb_20220919-cc37e9b8.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tanet/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb/tanet_imagenet-pretrained-r50_8xb6-1x1x16-50e_sthv1-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. The checkpoints for reference repo can be downloaded [here](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing).
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to

- [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md)
- [Something-something V1](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv1/README.md)

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TANet model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TANet model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tanet/tanet_imagenet-pretrained-r50_8xb8-dense-1x1x8-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{liu2020tam,
  title={TAM: Temporal Adaptive Module for Video Recognition},
  author={Liu, Zhaoyang and Wang, Limin and Wu, Wayne and Qian, Chen and Lu, Tong},
  journal={arXiv preprint arXiv:2005.06803},
  year={2020}
}
```

## TimeSformer

[Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named "TimeSformer," adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of frame-level patches. Our experimental study compares different self-attention schemes and suggests that "divided attention," where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered. Despite the radically new design, TimeSformer achieves state-of-the-art results on several action recognition benchmarks, including the best reported accuracy on Kinetics-400 and Kinetics-600. Finally, compared to 3D convolutional networks, our model is faster to train, it can achieve dramatically higher test efficiency (at a small drop in accuracy), and it can also be applied to much longer video clips (over one minute long).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018542-7f782ec9-dca2-495e-9043-c13ad941a25c.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus |        backbone         |   pretrain   | top1 acc | top5 acc | testing protocol | FLOPs | params |             config             |             ckpt             |             log             |
| :---------------------: | :--------: | :--: | :---------------------: | :----------: | :------: | :------: | :--------------: | :---: | :----: | :----------------------------: | :--------------------------: | :-------------------------: |
|         8x32x1          |  224x224   |  8   |   TimeSformer (divST)   | ImageNet-21K |  77.69   |  93.45   | 1 clip x 3 crop  | 196G  |  122M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.log) |
|         8x32x1          |  224x224   |  8   |  TimeSformer (jointST)  | ImageNet-21K |  76.95   |  93.28   | 1 clip x 3 crop  | 180G  | 86.11M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-8022d1c0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb.log) |
|         8x32x1          |  224x224   |  8   | TimeSformer (spaceOnly) | ImageNet-21K |  76.93   |  92.88   | 1 clip x 3 crop  | 141G  | 86.11M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb_20220815-78f05367.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. We keep the test setting with the [original repo](https://github.com/facebookresearch/TimeSformer) (three crop x 1 clip).
3. The pretrained model `vit_base_patch16_224.pth` used by TimeSformer was converted from [vision_transformer](https://github.com/google-research/vision_transformer).

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TimeSformer model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TimeSformer model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?},
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## TIN

[Temporal Interlacing Network](https://ojs.aaai.org/index.php/AAAI/article/view/6872)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

For a long time, the vision community tries to learn the spatio-temporal representation by combining convolutional neural network together with various temporal models, such as the families of Markov chain, optical flow, RNN and temporal convolution. However, these pipelines consume enormous computing resources due to the alternately learning process for spatial and temporal information. One natural question is whether we can embed the temporal information into the spatial one so the information in the two domains can be jointly learned once-only. In this work, we answer this question by presenting a simple yet powerful operator -- temporal interlacing network (TIN). Instead of learning the temporal features, TIN fuses the two kinds of information by interlacing spatial representations from the past to the future, and vice versa. A differentiable interlacing target can be learned to control the interlacing process. In this way, a heavy temporal model is replaced by a simple interlacing operator. We theoretically prove that with a learnable interlacing target, TIN performs equivalently to the regularized temporal convolution network (r-TCN), but gains 4% more accuracy with 6x less latency on 6 challenging benchmarks. These results push the state-of-the-art performances of video understanding by a considerable margin. Not surprising, the ensemble model of the proposed TIN won the 1st place in the ICCV19 - Multi Moments in Time challenge.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018602-d32bd546-e4f5-442c-9173-e4303676efb3.png" width="800"/>
</div>

### Results and Models

#### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 100 | 8x4  | ResNet50 | ImageNet |  39.68   |  68.55   |       44.04        |       72.72        | 8 clips x 1 crop |            x            |    6181    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb_20220913-9b7804d6.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.log) |

#### Something-Something V2

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 240 | 8x4  | ResNet50 | ImageNet |  54.78   |  82.18   |       56.48        |       83.45        | 8 clips x 1 crop |            x            |    6185    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb_20220913-84f9b4b0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.log) |

#### Kinetics-400

| frame sampling strategy |   resolution   | gpus | backbone |    pretrain     | top1 acc | top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |          config           |          ckpt           |           log           |
| :---------------------: | :------------: | :--: | :------: | :-------------: | :------: | :------: | :--------------: | :---------------------: | :--------: | :-----------------------: | :---------------------: | :---------------------: |
|          1x1x8          | short-side 256 | 8x4  | ResNet50 | TSM-Kinetics400 |  71.77   |  90.36   | 8 clips x 1 crop |            x            |    6185    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb_20220913-7f10d0c0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb.log) |

Here, we use `finetune` to indicate that we use [TSM model](https://download.openmmlab.com/mmaction/v1.0/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth) trained on Kinetics-400 to finetune the TIN model on Kinetics-400.

:::{note}

1. The **reference topk acc** are got by training the [original repo ##1aacd0c](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) with no [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4).
   The [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4) will lead to incorrect performance, so we fix it before running.
2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Prepare Datasets](en/user_guides/2_data_prepare.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TIN model on Something-Something V1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py \
    --work-dir work_dirs/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TIN model on Something-Something V1 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.json
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```

## TPN

[Temporal Pyramid Network for Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Temporal_Pyramid_Network_for_Action_Recognition_CVPR_2020_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Visual tempo characterizes the dynamics and the temporal scale of an action. Modeling such visual tempos of different actions facilitates their recognition. Previous works often capture the visual tempo through sampling raw videos at multiple rates and constructing an input-level frame pyramid, which usually requires a costly multi-branch network to handle. In this work we propose a generic Temporal Pyramid Network (TPN) at the feature-level, which can be flexibly integrated into 2D or 3D backbone networks in a plug-and-play manner. Two essential components of TPN, the source of features and the fusion of features, form a feature hierarchy for the backbone so that it can capture action instances at various tempos. TPN also shows consistent improvements over other challenging baselines on several action recognition datasets. Specifically, when equipped with TPN, the 3D ResNet-50 with dense sampling obtains a 2% gain on the validation set of Kinetics-400. A further analysis also reveals that TPN gains most of its improvements on action classes that have large variances in their visual tempos, validating the effectiveness of TPN.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018779-1d2a398f-dbd3-405a-87e5-e188b61fcc86.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |   reference top1 acc    |   reference top5 acc    | testing protocol  | inference time(video/s) | gpu_mem(M) |    config    |    ckpt    |    log    |
| :---------------------: | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :---------------------: | :---------------: | :---------------------: | :--------: | :----------: | :--------: | :-------: |
|          8x8x1          | short-side 320 | 8x2  | ResNet50 |   None   |  74.20   |  91.48   |            x            |            x            | 10 clips x 3 crop |            x            |    6916    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-97d0835d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.log) |
|          8x8x1          | short-side 320 |  8   | ResNet50 | ImageNet |  76.74   |  92.57   | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | 10 clips x 3 crop |            x            |    6916    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn-slowonly_imagenet-pretrained-r50_8xb8-8x8x1-150e_kinetics400-rgb_20220913-fed3f4c1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.log) |

#### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 100 | 8x6  | ResNet50 |   TSM    |  48.98   |  78.91   |         x          |         x          | 8 clips x 3 crop |            x            |    8828    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb_20220913-d2f5c300.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tpn/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb/tpn-tsm_imagenet-pretrained-r50_8xb8-1x1x8-150e_sthv1-rgb.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TPN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    --work-dir work_dirs/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb [--validate --seed 0 --deterministic]
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TPN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tpn/tpn-slowonly_r50_8xb8-8x8x1-150e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```

## TRN

[Temporal Relational Reasoning in Videos](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Temporal relational reasoning, the ability to link meaningful transformations of objects or entities over time, is a fundamental property of intelligent species. In this paper, we introduce an effective and interpretable network module, the Temporal Relation Network (TRN), designed to learn and reason about temporal dependencies between video frames at multiple time scales. We evaluate TRN-equipped networks on activity recognition tasks using three recent video datasets - Something-Something, Jester, and Charades - which fundamentally depend on temporal relational reasoning. Our results demonstrate that the proposed TRN gives convolutional neural networks a remarkable capacity to discover temporal relations in videos. Through only sparsely sampled video frames, TRN-equipped networks can accurately predict human-object interactions in the Something-Something dataset and identify various human gestures on the Jester dataset with very competitive performance. TRN-equipped networks also outperform two-stream networks and 3D convolution networks in recognizing daily activities in the Charades dataset. Further analyses show that the models learn intuitive and interpretable visual common sense knowledge in videos.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018998-d2120c3d-a9a7-4e4c-90b1-1e5ff1fd5f06.png" width="800"/>
</div>

### Results and Models

#### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) |  testing protocol  | FLOPs  | params |        config         |        ckpt         |         log         |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :----------------: | :----: | :----: | :-------------------: | :-----------------: | :-----------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet |         31.60 / 33.65         |         60.15 / 62.22         | 16 clips x 10 crop | 42.94G | 26.64M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb_20220815-e13db2e9.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.log) |

#### Something-Something V2

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) |  testing protocol  | FLOPs  | params |        config         |        ckpt         |         log         |
| :---------------------: | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :----------------: | :----: | :----: | :-------------------: | :-----------------: | :-----------------: |
|          1x1x8          |  224x224   |  8   | ResNet50 | ImageNet |         47.65 / 51.20         |         76.27 / 78.42         | 16 clips x 10 crop | 42.94G | 26.64M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20220815-e01617db.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop only) and accurate setting (three crop and `twice_sample`).
3. In the original [repository](https://github.com/zhoubolei/TRN-pytorch), the author augments data with random flipping on something-something dataset, but the augmentation method may be wrong due to the direct actions, such as `push left to right`. So, we replaced `flip` with `flip with label mapping`, and change the testing method `TenCrop`, which has five flipped crops, to `Twice Sample & ThreeCrop`.
4. We use `ResNet50` instead of `BNInception` as the backbone of TRN. When Training `TRN-ResNet50` on sthv1 dataset in the original repository, we get top1 (top5) accuracy 30.542 (58.627) vs. ours 31.81 (60.47).

For more details on data preparation, you can refer to [Something-something V1](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv1/README.md) and [Something-something V2](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv2/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TRN model on sthv1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TRN model on sthv1 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/trn/trn_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv1-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```

## TSM

[TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

The explosive growth in video streaming gives rise to challenges on performing video understanding at high accuracy and low computation cost. Conventional 2D CNNs are computationally cheap but cannot capture temporal relationships; 3D CNN based methods can achieve good performance but are computationally intensive, making it expensive to deploy. In this paper, we propose a generic and effective Temporal Shift Module (TSM) that enjoys both high efficiency and high performance. Specifically, it can achieve the performance of 3D CNN but maintain 2D CNN's complexity. TSM shifts part of the channels along the temporal dimension; thus facilitate information exchanged among neighboring frames. It can be inserted into 2D CNNs to achieve temporal modeling at zero computation and zero parameters. We also extended TSM to online setting, which enables real-time low-latency online video recognition and video object detection. TSM is accurate and efficient: it ranks the first place on the Something-Something leaderboard upon publication; on Jetson Nano and Galaxy Note8, it achieves a low latency of 13ms and 35ms for online video recognition.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019083-abc0de39-9ea1-4175-be5c-073c90de64c3.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | gpus |           backbone            | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |            config            |                       ckpt |                        log |
| :---------------------: | :--------: | :--: | :---------------------------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :--------------------------: | -------------------------: | -------------------------: |
|          1x1x8          |  224x224   |  8   |           ResNet50            | ImageNet |  73.18   |  90.56   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   |           ResNet50            | ImageNet |  73.22   |  90.22   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb_20220831-a6db1e5d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb.log) |
|         1x1x16          |  224x224   |  8   |           ResNet50            | ImageNet |  75.12   |  91.55   | 16 clips x 10 crop | 65.75G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb_20220831-042b1748.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_kinetics400-rgb.log) |
|      1x1x8 (dense)      |  224x224   |  8   |           ResNet50            | ImageNet |  73.38   |  90.78   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb_20220831-f55d3c2b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-dense-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet50 (NonLocalDotProduct) | ImageNet |  74.49   |  91.15   | 8 clips x 10 crop  | 61.30G | 31.68M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb_20220831-108bfde5.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-dot-product_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   |   ResNet50 (NonLocalGauss)    | ImageNet |  73.66   |  90.99   | 8 clips x 10 crop  | 59.06G | 28.00M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb_20220831-7e54dacf.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  74.34   |  91.23   | 8 clips x 10 crop  | 61.30G | 31.68M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb_20220831-35eddb57.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50-nl-embedded-gaussian_8xb16-1x1x8-50e_kinetics400-rgb.log) |

#### Something-something V2

| frame sampling strategy | resolution | gpus | backbone  | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :-------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|          1x1x8          |  224x224   |  8   | ResNet50  | ImageNet |  60.20   |  86.13   | 8 clips x 10 crop  | 32.88G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb_20221122-446d261a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_sthv2-rgb.log) |
|         1x1x16          |  224x224   |  8   | ResNet50  | ImageNet |  62.46   |  87.75   | 16 clips x 10 crop | 65.75G | 23.87M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb_20221122-b1fb8264.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x16-50e_sthv2-rgb.log) |
|          1x1x8          |  224x224   |  8   | ResNet101 | ImageNet |  60.49   |  85.99   | 8 clips x 10 crop  | 62.66G | 42.86M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb_20221122-cb2cc64e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb/tsm_imagenet-pretrained-r101_8xb16-1x1x8-50e_sthv2-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py \
     --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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

## TSN

[Temporal segment networks: Towards good practices for deep action recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Deep convolutional networks have achieved great success for visual recognition in still images. However, for action recognition in videos, the advantage over traditional methods is not so evident. This paper aims to discover the principles to design effective ConvNet architectures for action recognition in videos and learn these models given limited training samples. Our first contribution is temporal segment network (TSN), a novel framework for video-based action recognition. which is based on the idea of long-range temporal structure modeling. It combines a sparse temporal sampling strategy and video-level supervision to enable efficient and effective learning using the whole action video. The other contribution is our study on a series of good practices in learning ConvNets on video data with the help of temporal segment network. Our approach obtains the state-the-of-art performance on the datasets of HMDB51 ( 69.4%) and UCF101 (94.2%). We also visualize the learned ConvNet models, which qualitatively demonstrates the effectiveness of temporal segment network and the proposed good practices.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019237-8823045b-dfa3-45cc-a992-ee83ab9d8459.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | scheduler | resolution | gpus | backbone  | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |              config              |                           ckpt |                           log |
| :---------------------: | :-------: | :--------: | :--: | :-------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :------------------------------: | -----------------------------: | ----------------------------: |
|          1x1x3          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  72.83   |  90.65   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x5          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  73.80   |  91.21   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb_20220906-65d68713.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  74.12   |  91.34   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.log) |
|       dense-1x1x5       | MultiStep |  224x224   |  8   | ResNet50  | ImageNet |  71.37   |  89.67   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb_20220906-dcbc6e01.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb.log) |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet101 | ImageNet |  75.89   |  92.07   | 25 clips x 10 crop | 195.8G | 43.32M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb_20220906-23cff032.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r101_8xb32-1x1x8-100e_kinetics400-rgb.log) |

#### Something-Something V2

| frame sampling strategy | scheduler | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |              config              |                           ckpt |                            log |
| :---------------------: | :-------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :------------------------------: | -----------------------------: | -----------------------------: |
|          1x1x8          | MultiStep |  224x224   |  8   | ResNet50 | ImageNet |  34.85   |  66.37   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb_20221122-ad2dbb37.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.log) |
|         1x1x16          | MultiStep |  224x224   |  8   | ResNet50 | ImageNet |  36.55   |  68.00   | 25 clips x 10 crop | 102.7G | 24.33M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb_20221122-ee13c8e2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb.log) |

#### Using backbones from 3rd-party in TSN

It's possible and convenient to use a 3rd-party backbone for TSN under the framework of MMAction2, here we provide some examples for:

- [x] Backbones from [MMClassification](https://github.com/open-mmlab/mmclassification/)
- [x] Backbones from [TorchVision](https://github.com/pytorch/vision/)
- [x] Backbones from [TIMM (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models)

| frame sampling strategy | scheduler | resolution | gpus |     backbone     | pretrain | top1 acc | top5 acc |  testing protocol  | FLOPs  | params |            config             |                         ckpt |                         log |
| :---------------------: | :-------: | :--------: | :--: | :--------------: | :------: | :------: | :------: | :----------------: | :----: | :----: | :---------------------------: | ---------------------------: | --------------------------: |
|          1x1x3          | MultiStep |  224x224   |  8   |    ResNext101    | ImageNet |  72.95   |  90.36   | 25 clips x 10 crop | 200.3G | 42.95M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb_20221209-de2d5615.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-rn101-32x4d_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x3          | MultiStep |  224x224   |  8   |   DenseNet161    | ImageNet |  72.07   |  90.15   | 25 clips x 10 crop | 194.6G | 27.36M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb_20220906-5f4c0daf.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-dense161_8xb32-1x1x3-100e_kinetics400-rgb.log) |
|          1x1x3          | MultiStep |  224x224   |  8   | Swin Transformer | ImageNet |  77.03   |  92.61   | 25 clips x 10 crop | 386.7G | 87.15M | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb_20220906-65ed814e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_8xb32-1x1x3-100e_kinetics400-rgb.log) |

1. Note that some backbones in TIMM are not supported due to multiple reasons. Please refer to to [PR ##880](https://github.com/open-mmlab/mmaction2/pull/880) for details.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to

- [Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md)
- [Something-something V2](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv2/README.md)

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py  \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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

## UniFormer

[UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning](https://arxiv.org/abs/2201.04676)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

It is a challenging task to learn rich and multi-scale spatiotemporal semantics from high-dimensional videos, due to large local redundancy and complex global dependency between video frames. The recent advances in this research have been mainly driven by 3D convolutional neural networks and vision transformers. Although 3D convolution can efficiently aggregate local context to suppress local redundancy from a small 3D neighborhood, it lacks the capability to capture global dependency because of the limited receptive field. Alternatively, vision transformers can effectively capture long-range dependency by self-attention mechanism, while having the limitation on reducing local redundancy with blind similarity comparison among all the tokens in each layer. Based on these observations, we propose a novel Unified transFormer (UniFormer) which seamlessly integrates merits of 3D convolution and spatiotemporal self-attention in a concise transformer format, and achieves a preferable balance between computation and accuracy. Different from traditional transformers, our relation aggregator can tackle both spatiotemporal redundancy and dependency, by learning local and global token affinity respectively in shallow and deep layers. We conduct extensive experiments on the popular video benchmarks, e.g., Kinetics-400, Kinetics-600, and Something-Something V1&V2. With only ImageNet-1K pretraining, our UniFormer achieves 82.9%/84.8% top-1 accuracy on Kinetics-400/Kinetics-600, while requiring 10x fewer GFLOPs than other state-of-the-art methods. For Something-Something V1 and V2, our UniFormer achieves new state-of-the-art performances of 60.9% and 71.2% top-1 accuracy respectively.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/Sense-X/UniFormer/main/figures/framework.png"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy |   resolution   |  backbone   | top1 acc | top5 acc | [reference](<(https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md)>) top1 acc | [reference](<(https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md)>) top5 acc | mm-Kinetics top1 acc | mm-Kinetics top5 acc | testing protocol | FLOPs | params |                                                                        config                                                                        |                                                                           ckpt                                                                           |
| :---------------------: | :------------: | :---------: | :------: | :------: | :-----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: | :------------------: | :------------------: | :--------------: | :---: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|         16x4x1          | short-side 320 | UniFormer-S |   80.9   |   94.6   |                                                  80.8                                                   |                                                  94.7                                                   |         80.9         |         94.6         | 4 clips x 1 crop | 41.8G | 21.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformer/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-c630a037.pth) |
|         16x4x1          | short-side 320 | UniFormer-B |   82.0   |   95.0   |                                                  82.0                                                   |                                                  95.1                                                   |         82.0         |         95.0         | 4 clips x 1 crop | 96.7G | 49.8M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformer/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-157c2e66.pth)  |
|         32x4x1          | short-side 320 | UniFormer-B |   83.1   |   95.3   |                                                  82.9                                                   |                                                  95.4                                                   |         83.0         |         95.3         | 4 clips x 1 crop |  59G  | 49.8M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformer/uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb_20221219-b776322c.pth)  |

The models are ported from the repo [UniFormer](https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md) and tested on our data. Currently, we only support the testing of UniFormer models, training will be available soon.

1. The values in columns named after "reference" are the results of the original repo.
2. The values in `top1/5 acc` is tested on the same data list as the original repo, and the label map is provided by [UniFormer](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL). The total videos are available at [Kinetics400](https://pan.baidu.com/s/1t5K0FRz3PGAT-37-3FwAfg) (BaiduYun password: g5kp), which consists of 19787 videos.
3. The values in columns named after "mm-Kinetics" are the testing results on the Kinetics dataset held by MMAction2, which is also used by other models in MMAction2. Due to the differences between various versions of Kinetics dataset, there is a little gap between `top1/5 acc` and `mm-Kinetics top1/5 acc`. For a fair comparison with other models, we report both results here. Note that we simply report the inference results, since the training set is different between UniFormer and other models, the results are lower than that tested on the author's version.
4. Since the original models for Kinetics-400/600/700 adopt different [label file](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL), we simply map the weight according to the label name. New label map for Kinetics-400/600/700 can be found [here](https://github.com/open-mmlab/mmaction2/tree/dev-1.x/tools/data/kinetics).
5. Due to some difference between [SlowFast](https://github.com/facebookresearch/SlowFast) and MMAction, there are some gaps between their performances.

For more details on data preparation, you can refer to [preparing_kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test UniFormer-S model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/uniformer/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{
  li2022uniformer,
  title={UniFormer: Unified Transformer for Efficient Spatial-Temporal Representation Learning},
  author={Kunchang Li and Yali Wang and Gao Peng and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=nBU_u6DLvoK}
}
```

## UniFormerV2

[UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer](https://arxiv.org/abs/2211.09552)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Learning discriminative spatiotemporal representation is the key problem of video understanding. Recently, Vision Transformers (ViTs) have shown their power in learning long-term video dependency with self-attention. Unfortunately, they exhibit limitations in tackling local video redundancy, due to the blind global comparison among tokens. UniFormer has successfully alleviated this issue, by unifying convolution and self-attention as a relation aggregator in the transformer format. However, this model has to require a tiresome and complicated image-pretraining phrase, before being finetuned on videos. This blocks its wide usage in practice. On the contrary, open-sourced ViTs are readily available and well-pretrained with rich image supervision. Based on these observations, we propose a generic paradigm to build a powerful family of video networks, by arming the pretrained ViTs with efficient UniFormer designs. We call this family UniFormerV2, since it inherits the concise style of the UniFormer block. But it contains brand-new local and global relation aggregators, which allow for preferable accuracy-computation balance by seamlessly integrating advantages from both ViTs and UniFormer. Without any bells and whistles, our UniFormerV2 gets the state-of-the-art recognition performance on 8 popular video benchmarks, including scene-related Kinetics-400/600/700 and Moments in Time, temporal-related Something-Something V1/V2, untrimmed ActivityNet and HACS. In particular, it is the first model to achieve 90% top-1 accuracy on Kinetics-400, to our best knowledge.

<!-- [IMAGE] -->

<div align=center>
<img src="https://raw.githubusercontent.com/OpenGVLab/UniFormerV2/main/img/framework.png"/>
</div>

### Results and Models

#### Kinetics-400

| uniform sampling |   resolution   |       backbone       | top1 acc | top5 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top1 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top5 acc | mm-Kinetics top1 acc | mm-Kinetics top5 acc | testing protocol | FLOPs | params |                                                                                 config                                                                                 |                                                                                         ckpt                                                                                         |
| :--------------: | :------------: | :------------------: | :------: | :------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :------------------: | :------------------: | :--------------: | :---: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        8         | short-side 320 |   UniFormerV2-B/16   |   85.8   |   97.1   |                                           85.6                                            |                                           97.0                                            |         85.8         |         97.1         | 4 clips x 3 crop | 0.1T  |  115M  |  [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics400-rgb.py)  |  [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics400/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics400-rgb_20221219-203d6aac.pth)  |
|        8         | short-side 320 |   UniFormerV2-L/14   |   88.7   |   98.1   |                                           88.8                                            |                                           98.1                                            |         88.7         |         98.1         | 4 clips x 3 crop | 0.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics400-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics400/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics400-rgb_20221219-972ea063.pth)  |
|        16        | short-side 320 |   UniFormerV2-L/14   |   89.0   |   98.2   |                                           89.1                                            |                                           98.2                                            |         89.0         |         98.2         | 4 clips x 3 crop | 1.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics400/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics400-rgb_20221219-6dc86d05.pth) |
|        32        | short-side 320 |   UniFormerV2-L/14   |   89.3   |   98.2   |                                           89.3                                            |                                           98.2                                            |         89.4         |         98.2         | 2 clips x 3 crop | 2.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics400/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics400-rgb_20221219-56a46f64.pth) |
|        32        | short-side 320 | UniFormerV2-L/14@336 |   89.5   |   98.4   |                                           89.7                                            |                                           98.3                                            |         89.5         |         98.4         | 2 clips x 3 crop | 6.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics400/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics400-rgb_20221219-1dd7650f.pth) |

#### Kinetics-600

| uniform sampling | resolution |       backbone       | top1 acc | top5 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top1 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top5 acc | mm-Kinetics top1 acc | mm-Kinetics top5 acc | testing protocol | FLOPs | params |                                                                                 config                                                                                 |                                                                                         ckpt                                                                                         |
| :--------------: | :--------: | :------------------: | :------: | :------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :------------------: | :------------------: | :--------------: | :---: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        8         |    Raw     |   UniFormerV2-B/16   |   86.4   |   97.3   |                                           86.1                                            |                                           97.2                                            |         85.5         |         97.0         | 4 clips x 3 crop | 0.1T  |  115M  |  [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics600-rgb.py)  |  [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics600/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics600-rgb_20221219-c62c4da4.pth)  |
|        8         |    Raw     |   UniFormerV2-L/14   |   89.0   |   98.3   |                                           89.0                                            |                                           98.2                                            |         87.5         |         98.0         | 4 clips x 3 crop | 0.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics600-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics600/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics600-rgb_20221219-cf88e4c2.pth)  |
|        16        |    Raw     |   UniFormerV2-L/14   |   89.4   |   98.3   |                                           89.4                                            |                                           98.3                                            |         87.8         |         98.0         | 4 clips x 3 crop | 1.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics600-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics600/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics600-rgb_20221219-38ff0e3e.pth) |
|        32        |    Raw     |   UniFormerV2-L/14   |   89.2   |   98.3   |                                           89.5                                            |                                           98.3                                            |         87.7         |         98.1         | 2 clips x 3 crop | 2.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics600-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics600/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-d450d071.pth) |
|        32        |    Raw     | UniFormerV2-L/14@336 |   89.8   |   98.5   |                                           89.9                                            |                                           98.5                                            |         88.8         |         98.3         | 2 clips x 3 crop | 6.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics600/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics600-rgb_20221219-f984f5d2.pth) |

#### Kinetics-700

| uniform sampling | resolution |       backbone       | top1 acc | top5 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top1 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top5 acc | mm-Kinetics top1 acc | mm-Kinetics top5 acc | testing protocol | FLOPs | params |                                                                                 config                                                                                 |                                                                                         ckpt                                                                                         |
| :--------------: | :--------: | :------------------: | :------: | :------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :------------------: | :------------------: | :--------------: | :---: | :----: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        8         |    Raw     |   UniFormerV2-B/16   |   76.3   |   92.9   |                                           76.3                                            |                                           92.7                                            |         75.1         |         92.5         | 4 clips x 3 crop | 0.1T  |  115M  |  [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics700-rgb.py)  |  [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics700/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics700-rgb_20221219-8a7c4ac4.pth)  |
|        8         |    Raw     |   UniFormerV2-L/14   |   80.8   |   95.2   |                                           80.8                                            |                                           95.4                                            |         79.4         |         94.8         | 4 clips x 3 crop | 0.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics700-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics700/uniformerv2-large-p14-res224_clip-kinetics710-pre_u8_kinetics700-rgb_20221219-bfb9f401.pth)  |
|        16        |    Raw     |   UniFormerV2-L/14   |   81.2   |   95.6   |                                           81.2                                            |                                           95.6                                            |         79.2         |         95.0         | 4 clips x 3 crop | 1.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics700/uniformerv2-large-p14-res224_clip-kinetics710-pre_u16_kinetics700-rgb_20221219-745209d2.pth) |
|        32        |    Raw     |   UniFormerV2-L/14   |   81.4   |   95.7   |                                           81.5                                            |                                           95.7                                            |         79.8         |         95.3         | 2 clips x 3 crop | 2.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics700/uniformerv2-large-p14-res224_clip-kinetics710-pre_u32_kinetics700-rgb_20221219-eebe7056.pth) |
|        32        |    Raw     | UniFormerV2-L/14@336 |   82.1   |   96.0   |                                           82.1                                            |                                           96.1                                            |         80.6         |         95.6         | 2 clips x 3 crop | 6.3T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics700/uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics700-rgb_20221219-95cf9046.pth) |

#### MiTv1

| uniform sampling | resolution |       backbone       | top1 acc | top5 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top1 acc | [reference](<(https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md)>) top5 acc | testing protocol | FLOPs | params |                                                                                    config                                                                                     |                                                                                         ckpt                                                                                          |
| :--------------: | :--------: | :------------------: | :------: | :------: | :---------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: | :--------------: | :---: | :----: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|        8         |    Raw     |   UniFormerV2-B/16   |   42.7   |   71.6   |                                           42.6                                            |                                           71.7                                            | 4 clips x 3 crop | 0.1T  |  115M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/mitv1/uniformerv2-base-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb_20221219-fddbc786.pth)  |
|        8         |    Raw     |   UniFormerV2-L/14   |   47.0   |   76.1   |                                           47.0                                            |                                           76.1                                            | 4 clips x 3 crop | 0.7T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/mitv1/uniformerv2-large-p16-res224_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb_20221219-882c0598.pth) |
|        8         |    Raw     | UniFormerV2-L/14@336 |   47.7   |   76.8   |                                           47.8                                            |                                           76.0                                            | 4 clips x 3 crop | 1.6T  |  354M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/mitv1/uniformerv2-large-p16-res336_clip-kinetics710-kinetics-k400-pre_u8_mitv1-rgb_20221219-9020986e.pth) |

#### Kinetics-710

| uniform sampling | resolution |       backbone       |                                     config                                     |                                     ckpt                                     |
| :--------------: | :--------: | :------------------: | :----------------------------------------------------------------------------: | :--------------------------------------------------------------------------: |
|        8         |    Raw     |   UniFormerV2-B/16   | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20221219-77d34f81.pth) |
|        8         |    Raw     |   UniFormerV2-L/14   | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res224_clip-pre_u8_kinetics710-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-large-p14-res224_clip-pre_u8_kinetics710-rgb_20221219-bfaae587.pth) |
|        8         |    Raw     | UniFormerV2-L/14@336 | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/uniformerv2/uniformerv2-large-p14-res336_clip-pre_u8_kinetics710-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/kinetics710/uniformerv2-large-p14-res336_clip-pre_u8_kinetics710-rgb_20221219-55878cdc.pth) |

The models are ported from the repo [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2/blob/main/MODEL_ZOO.md) and tested on our data. Currently, we only support the testing of UniFormerV2 models, training will be available soon.

1. The values in columns named after "reference" are the results of the original repo.
2. The values in `top1/5 acc` is tested on the same data list as the original repo, and the label map is provided by [UniFormerV2](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL).
3. The values in columns named after "mm-Kinetics" are the testing results on the Kinetics dataset held by MMAction2, which is also used by other models in MMAction2. Due to the differences between various versions of Kinetics dataset, there is a little gap between `top1/5 acc` and `mm-Kinetics top1/5 acc`. For a fair comparison with other models, we report both results here. Note that we simply report the inference results, since the training set is different between UniFormer and other models, the results are lower than that tested on the author's version.
4. Since the original models for Kinetics-400/600/700 adopt different [label file](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL), we simply map the weight according to the label name. New label map for Kinetics-400/600/700 can be found [here](https://github.com/open-mmlab/mmaction2/tree/dev-1.x/tools/data/kinetics).
5. Due to some differences between [SlowFast](https://github.com/facebookresearch/SlowFast) and MMAction2, there are some gaps between their performances.
6. Kinetics-710 is used for pretraining, which helps improve the performance on other datasets efficiently. You can find more details in the [paper](https://arxiv.org/abs/2211.09552).

For more details on data preparation, you can refer to

- [preparing_kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md)
- [preparing_mit](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/mit/README.md)

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test UniFormerV2-B/16 model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-kinetics710-pre_u8_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{Li2022UniFormerV2SL,
  title={UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer},
  author={Kunchang Li and Yali Wang and Yinan He and Yizhuo Li and Yi Wang and Limin Wang and Y. Qiao},
  journal={ArXiv},
  year={2022},
  volume={abs/2211.09552}
}
```

## VideoMAE

[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking with an extremely high ratio. This simple design makes video reconstruction a more challenging self-supervision task, thus encouraging extracting more effective video representations during this pre-training process. We obtain three important findings on SSVP: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables a higher masking ratio than that of images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain shift between pre-training and target datasets is an important issue. Notably, our VideoMAE with the vanilla ViT can achieve 87.4% on Kinetics-400, 75.4% on Something-Something V2, 91.3% on UCF101, and 62.6% on HMDB51, without using any extra data.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/35267818/191656296-14f28f4a-203f-4eeb-a4c3-c2efdb6d1ab4.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy |   resolution   | backbone | top1 acc | top5 acc |         reference top1 acc         |         reference top5 acc         | testing protocol  | FLOPs | params |         config         |         ckpt          |
| :---------------------: | :------------: | :------: | :------: | :------: | :--------------------------------: | :--------------------------------: | :---------------: | :---: | :----: | :--------------------: | :-------------------: |
|         16x4x1          | short-side 320 |  ViT-B   |   81.3   |   95.0   | 81.5 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 95.1 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops | 180G  |  87M   | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth) \[1\] |
|         16x4x1          | short-side 320 |  ViT-L   |   85.3   |   96.7   | 85.2 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 96.8 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops | 597G  |  305M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) \[1\] |

\[1\] The models are ported from the repo [VideoMAE](https://github.com/MCG-NJU/VideoMAE) and tested on our data. Currently, we only support the testing of VideoMAE models, training will be available soon.

1. The values in columns named after "reference" are the results of the original repo.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [preparing_kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ViT-base model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## X3D

[X3D: Expanding Architectures for Efficient Video Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

This paper presents X3D, a family of efficient video networks that progressively expand a tiny 2D image classification architecture along multiple network axes, in space, time, width and depth. Inspired by feature selection methods in machine learning, a simple stepwise network expansion approach is employed that expands a single axis in each step, such that good accuracy to complexity trade-off is achieved. To expand X3D to a specific target complexity, we perform progressive forward expansion followed by backward contraction. X3D achieves state-of-the-art performance while requiring 4.8x and 5.5x fewer multiply-adds and parameters for similar accuracy as previous work. Our most surprising finding is that networks with high spatiotemporal resolution can perform well, while being extremely light in terms of network width and parameters. We report competitive accuracy at unprecedented efficiency on video classification and detection benchmarks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019391-6711febb-9e5d-4bec-85b9-65f5179e93a2.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | resolution | backbone | top1 10-view | top1 30-view |           reference top1 10-view           |           reference top1 30-view           |           config           |           ckpt            |
| :---------------------: | :--------: | :------: | :----------: | :----------: | :----------------------------------------: | :----------------------------------------: | :------------------------: | :-----------------------: |
|         13x6x1          |  160x160   |  X3D_S   |     73.2     |     73.3     | 73.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 73.5 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/x3d/facebook/x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth)\[1\] |
|         16x5x1          |  224x224   |  X3D_M   |     75.2     |     76.4     | 75.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 76.2 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/x3d/facebook/x3d_m_16x5x1_facebook-kinetics400-rgb_20201027-3f42382a.pth)\[1\] |

\[1\] The models are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3D models, training will be available soon.

1. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
2. The validation set of Kinetics400 we used is same as the repo [SlowFast](https://github.com/facebookresearch/SlowFast/), which is available [here](https://github.com/facebookresearch/video-nonlocal-net/issues/67).

For more details on data preparation, you can refer to [Kinetics400](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test X3D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@misc{feichtenhofer2020x3d,
      title={X3D: Expanding Architectures for Efficient Video Recognition},
      author={Christoph Feichtenhofer},
      year={2020},
      eprint={2004.04730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## ResNet for Audio

[Audiovisual SlowFast Networks for Video Recognition](https://arxiv.org/abs/2001.08740)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present Audiovisual SlowFast Networks, an architecture for integrated audiovisual perception. AVSlowFast has Slow and Fast visual pathways that are deeply inte- grated with a Faster Audio pathway to model vision and sound in a unified representation. We fuse audio and vi- sual features at multiple layers, enabling audio to con- tribute to the formation of hierarchical audiovisual con- cepts. To overcome training difficulties that arise from dif- ferent learning dynamics for audio and visual modalities, we introduce DropPathway, which randomly drops the Au- dio pathway during training as an effective regularization technique. Inspired by prior studies in neuroscience, we perform hierarchical audiovisual synchronization to learn joint audiovisual features. We report state-of-the-art results on six video action classification and detection datasets, perform detailed ablation studies, and show the gener- alization of AVSlowFast to learn self-supervised audiovi- sual features. Code will be made available at: https: //github.com/facebookresearch/SlowFast.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/30782254/147050415-a30ad32a-ce52-452d-ac3d-91058c8d0cc9.png" width="800"/>
</div>

### Results and Models

#### Kinetics-400

| frame sampling strategy | n_fft | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol | gpu_mem(M) |                 config                 |                 ckpt                 |                 log                  |
| :---------------------: | :---: | :--: | :------: | :------: | :------: | :------: | :--------------: | :--------: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
|         64x1x1          | 1024  |  8   | Resnet18 |   None   |   19.7   |  35.75   |     10 clips     |    1897    | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature_20201012-bf34df6c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to `Prepare audio` in [Data Preparation Tutorial](en/user_guides/2_data_prepare.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ResNet model on Kinetics-400 audio dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ResNet model on Kinetics-400 audio dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@article{xiao2020audiovisual,
  title={Audiovisual SlowFast Networks for Video Recognition},
  author={Xiao, Fanyi and Lee, Yong Jae and Grauman, Kristen and Malik, Jitendra and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2001.08740},
  year={2020}
}
```
