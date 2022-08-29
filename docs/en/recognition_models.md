# Action Recognition Models

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

| config                               | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | inference_time(video/s) | gpu_mem(M) |                ckpt                |                log                 |
| :----------------------------------- | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :---------------------: | :--------: | :--------------------------------: | :--------------------------------: |
| [c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb.py](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb.py) |  128x171   |  8   |   c3d    | sports1m |  82.92   |  96.11   | 10 clips x 1 crop |            x            |    6067    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb_20220811-31723200.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb/20220722_140343.log) |

:::{note}

1. The author of C3D normalized UCF-101 with volume mean and used SVM to classify videos, while we normalized the dataset with RGB mean value and used a linear classifier.
2. The **gpus** indicates the number of gpu (80G A100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
3. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.

:::

For more details on data preparation, you can refer to UCF-101 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C3D model on UCF-101 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C3D model on UCF-101 dataset.

```shell
python tools/test.py configs/recognition/c3d/c3d_sports1m_16x1x1_45e_8xb30_ucf101_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                   |   resolution   | gpus | backbone  | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                  |                  log                   |
| :--------------------------------------- | :------------: | :--: | :-------: | :------: | :------: | :------: | :---------------------: | :--------: | :------------------------------------: | :------------------------------------: |
| [ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb.py) | short-side 320 |  8   | ResNet152 |  IG65M   |  82.66   |  95.82   |            x            |   32703    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb_20220811-c7a3cc5b.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb/20220729_134436.log) |
| [ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb.py) | short-side 320 |  8   | ResNet152 |  IG65M   |  82.58   |  95.76   |            x            |   32703    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb_20220811-7d1dacde.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb/20220729_120335.log) |
| [ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_8xb12_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_8xb12_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50  |  IG65M   |  79.17   |  94.14   |            x            |   22238    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_8xb12_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_8xb12_kinetics400_rgb_20220811-44395bae.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_8xb12_kinetics400_rgb/20220731_014145.log) |
| [ipcsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb.py) | short-side 320 |  x   | ResNet152 |   None   |  77.69   |  92.83   |            x            |     x      | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-d565828d.pth) |                   x                    |
| [ircsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb.py) | short-side 320 |  x   | ResNet152 |   None   |  79.17   |  94.14   |            x            |     x      | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-5c933ae1.pth) |                   x                    |
| [ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py) | short-side 320 |  x   | ResNet152 |  IG65M   |  82.51   |  95.52   |            x            |     x      | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth) |                   x                    |
| [ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py) | short-side 320 |  x   | ResNet152 | Sports1M |  78.77   |  93.78   |            x            |     x      | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth) |                   x                    |
| [ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/csn/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py) | short-side 320 |  x   | ResNet152 | Sports1M |  78.82   |  93.34   |            x            |     x      | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-b9b10241.pth) |                   x                    |

:::{note}

1. The **gpus** indicates the number of gpu (80G A100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. The **infer_ckpt** means those checkpoints are ported from [VMZ](https://github.com/facebookresearch/VMZ).

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb.py  \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CSN model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_8xb12_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [i3d_nl_dot_product_r50_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.76   |  91.84   |            x            |    6245    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_8xb8_kinetics400_rgb/i3d_nl_dot_product_r50_32x2x1_100e_8xb8_kinetics400_rgb_20220812-8e1f2148.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_8xb8_kinetics400_rgb/20220627_172159.log) |
| [i3d_nl_embedded_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.69   |  91.69   |            x            |    6415    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb/i3d_nl_embedded_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb_20220812-afd8f562.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb/20220629_135933.log) |
| [i3d_nl_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.90   |  91.15   |            x            |    6108    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb/i3d_nl_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb_20220812-0c5cbf5a.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_8xb8_kinetics400_rgb/20220722_135616.log) |
| [i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.22   |  91.11   |            x            |    5149    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb_20220812-e213c223.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb/20220627_165806.log) |
| [i3d_r50_dense_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.77   |  91.35   |            x            |    5151    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_dense_32x2x1_100e_8xb8_kinetics400_rgb/i3d_r50_dense_32x2x1_100e_8xb8_kinetics400_rgb_20220812-9f46003f.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_dense_32x2x1_100e_8xb8_kinetics400_rgb/20220627_172844.log) |
| [i3d_r50_heavy_32x2x1_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/i3d/i3d_r50_heavy_32x2x1_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  76.08   |  92.34   |            x            |   17350    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_heavy_32x2x1_100e_8xb8_kinetics400_rgb/i3d_r50_heavy_32x2x1_100e_8xb8_kinetics400_rgb_20220812-ed501b31.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/i3d/i3d_r50_heavy_32x2x1_100e_8xb8_kinetics400_rgb/20220722_000847.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train I3D model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/i3d/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/i3d/i3d_r50_32x2x1_100e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet34 |   None   |  69.35   |  88.32   |            x            |    5036    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb_20220812-47cfe041.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb/20220625_103729.log) |
| [r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb.py](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet34 |   None   |  75.27   |  92.03   |            x            |   17006    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb_20220812-4270588c.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb/20220726_172404.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                               |   resolution   | gpus |       backbone       | pretrain | top1 acc | top5 acc | inference_time(video/s)  | gpu_mem(M) |                ckpt                |                log                 |
| :----------------------------------- | :------------: | :--: | :------------------: | :------: | :------: | :------: | :----------------------: | :--------: | :--------------------------------: | :--------------------------------: |
| [slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   |       ResNet50       |   None   |  75.51   |  92.05   |            x             |    6331    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/20200731_151706.log) |
| [slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   |       ResNet50       |   None   |  76.24   |   92.3   | 1.3 ((32+8)x10x3 frames) |    9201    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/20200716_192653.log) |
| [slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb_steplr](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb_steplr.py) | short-side 320 |  8   |       ResNet50       |   None   |  75.72   |  92.37   |            x             |    9401    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr-43988bac.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr.log) |
| [slowfast_r101_8x8x1_256e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r101_8x8x1_256e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   |      ResNet101       |   None   |  76.29   |  91.97   |            x             |   13454    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/20210218_121513.log) |
| [slowfast_r101_r50_4x16x1_256e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowfast/slowfast_r101_r50_4x16x1_256e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | ResNet101 + ResNet50 |   None   |  76.33   |  93.03   |            x             |    8018    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/20210118_133528.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                          |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |             ckpt              |             log              |             json              |
| :------------------------------ | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :---------------------------: | :--------------------------: | :---------------------------: |
| [slowonly_r50_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |  72.76   |  90.51   |            x            |    3168    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json) |
| [slowonly_r50_video_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 |   None   |  72.90   |  90.82   |            x            |    8472    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014.json) |
| [slowonly_r50_8x8x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |  74.42   |  91.49   |            x            |    5820    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb_20200820-75851a7d.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_8x8x1_256e_kinetics400_rgb/20200817_003320.log.json) |
| [slowonly_r50_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 |   None   |  73.02   |  90.77   |    4.0 (40x3 frames)    |    3168    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
| [slowonly_r50_8x8x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py) | short-side 320 | 8x3  | ResNet50 |   None   |  74.93   |  91.92   |    2.3 (80x3 frames)    |    5820    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8x1_256e_kinetics400_rgb_20200703-a79c555a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/so_8x8.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb/slowonly_r50_8x8_74.93_91.92.log.json) |
| [slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 | ImageNet |  73.39   |  91.12   |            x            |    3168    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912-1e8fc736.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_4x16x1_150e_kinetics400_rgb_20200912.json) |
| [slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py) | short-side 320 | 8x4  | ResNet50 | ImageNet |  75.55   |  92.04   |            x            |    5820    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912.json) |
| [slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 | ImageNet |  74.54   |  91.73   |            x            |    4435    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb_20210308-0d6e5a69.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/20210305_152630.log.json) |
| [slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb.py) | short-side 320 | 8x4  | ResNet50 | ImageNet |  76.07   |  92.42   |            x            |    8895    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb_20210308-e8dd9e82.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb/20210308_212250.log.json) |
| [slowonly_r50_4x16x1_256e_kinetics400_flow](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow.py) | short-side 320 | 8x2  | ResNet50 | ImageNet |  61.79   |  83.62   |            x            |    8450    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_flow/slowonly_r50_4x16x1_256e_kinetics400_flow_61.8_83.6.log.json) |
| [slowonly_r50_8x8x1_196e_kinetics400_flow](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py) | short-side 320 | 8x4  | ResNet50 | ImageNet |  65.76   |  86.25   |            x            |    8455    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_196e_kinetics400_flow_65.8_86.3.log.json) |

#### Kinetics-400 Data Benchmark

In data benchmark, we compare two different data preprocessing methods: (1) Resize video to 340x256, (2) Resize the short edge of video to 320px, (3) Resize the short edge of video to 256px.

| config                            |   resolution   | gpus | backbone | Input | pretrain | top1 acc | top5 acc |  testing protocol  |              ckpt               |               log               |               json               |
| :-------------------------------- | :------------: | :--: | :------: | :---: | :------: | :------: | :------: | :----------------: | :-----------------------------: | :-----------------------------: | :------------------------------: |
| [slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb.py) |    340x256     | 8x2  | ResNet50 | 4x16  |   None   |  71.61   |  90.05   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803-dadca1a3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb/slowonly_r50_randomresizedcrop_340x256_4x16x1_256e_kinetics400_rgb_20200803.json) |
| [slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_320p_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | 8x2  | ResNet50 | 4x16  |   None   |  73.02   |  90.77   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
| [slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/data_benchmark/slowonly_r50_randomresizedcrop_256p_4x16x1_256e_kinetics400_rgb.py) | short-side 256 | 8x4  | ResNet50 | 4x16  |   None   |  72.76   |  90.51   | 10 clips x 3 crops | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb_20200820-bea7701f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_256p_4x16x1_256e_kinetics400_rgb/20200817_001411.log.json) |

#### Kinetics-400 OmniSource Experiments

|               config                |   resolution   | backbone  | pretrain |   w. OmniSource    | top1 acc | top5 acc |               ckpt                |                log                |                json                |
| :---------------------------------: | :------------: | :-------: | :------: | :----------------: | :------: | :------: | :-------------------------------: | :-------------------------------: | :--------------------------------: |
| [slowonly_r50_4x16x1_256e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py) | short-side 320 | ResNet50  |   None   |        :x:         |   73.0   |   90.8   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16x1_256e_kinetics400_rgb_20200704-a69556c6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/so_4x16.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb/slowonly_r50_4x16_73.02_90.77.log.json) |
|                  x                  |       x        | ResNet50  |   None   | :heavy_check_mark: |   76.8   |   92.5   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r50_omni_4x16x1_kinetics400_rgb_20200926-51b1f7ea.pth) |                 x                 |                 x                  |
| [slowonly_r101_8x8x1_196e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r101_8x8x1_196e_kinetics400_rgb.py) |       x        | ResNet101 |   None   |        :x:         |   76.5   |   92.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_without_omni_8x8x1_kinetics400_rgb_20200926-0c730aef.pth) |                 x                 |                 x                  |
|                  x                  |       x        | ResNet101 |   None   | :heavy_check_mark: |   80.4   |   94.4   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/omni/slowonly_r101_omni_8x8x1_kinetics400_rgb_20200926-b5dbb701.pth) |                 x                 |                 x                  |

#### Kinetics-600

| config                                  |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |                 ckpt                  |                 log                  |                  json                  |
| :-------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :-----------------------------------: | :----------------------------------: | :------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics600_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |   77.5   |   93.7   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015-81e5153e.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics600_rgb/slowonly_r50_video_8x8x1_256e_kinetics600_rgb_20201015.json) |

#### Kinetics-700

| config                                  |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |                 ckpt                  |                 log                  |                  json                  |
| :-------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :-----------------------------------: | :----------------------------------: | :------------------------------------: |
| [slowonly_r50_video_8x8x1_256e_kinetics700_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb.py) | short-side 256 | 8x4  | ResNet50 |   None   |   65.0   |   86.1   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015-9250f662.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_8x8x1_256e_kinetics700_rgb/slowonly_r50_video_8x8x1_256e_kinetics700_rgb_20201015.json) |

#### GYM99

| config                                |   resolution   | gpus | backbone | pretrain | top1 acc | mean class acc |                 ckpt                 |                 log                 |                 json                 |
| :------------------------------------ | :------------: | :--: | :------: | :------: | :------: | :------------: | :----------------------------------: | :---------------------------------: | :----------------------------------: |
| [slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb.py) | short-side 256 | 8x2  | ResNet50 | ImageNet |   79.3   |      70.2      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111-a9c34b54.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb/slowonly_imagenet_pretrained_r50_4x16x1_120e_gym99_rgb_20201111.json) |
| [slowonly_k400_pretrained_r50_4x16x1_120e_gym99_flow](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_k400_pretrained_r50_4x16x1_120e_gym99_flow.py) | short-side 256 | 8x2  | ResNet50 | Kinetics |   80.3   |      71.0      | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111-66ecdb3c.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow/slowonly_kinetics_pretrained_r50_4x16x1_120e_gym99_flow_20201111.json) |
| 1: 1 Fusion                           |                |      |          |          |   83.7   |      74.8      |                                      |                                     |                                      |

#### Jester

| config                                     | resolution | gpus | backbone | pretrain | top1 acc |                   ckpt                   |                   log                   |                   json                    |
| :----------------------------------------- | :--------: | :--: | :------: | :------: | :------: | :--------------------------------------: | :-------------------------------------: | :---------------------------------------: |
| [slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.py) | height 100 |  8   | ResNet50 | ImageNet |   97.2   | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb-b56a5389.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb/slowonly_imagenet_pretrained_r50_8x8x1_64e_jester_rgb.json) |

#### HMDB51

| config                                  | gpus | backbone |  pretrain   | top1 acc | top5 acc | gpu_mem(M) |                 ckpt                  |                  log                  |                  json                  |
| :-------------------------------------- | :--: | :------: | :---------: | :------: | :------: | :--------: | :-----------------------------------: | :-----------------------------------: | :------------------------------------: |
| [slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb.py) |  8   | ResNet50 |  ImageNet   |  37.52   |  71.50   |    5812    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb/slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb_20210630-16faeb6a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb/20210605_185256.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_hmdb51_rgb/20210605_185256.log.json) |
| [slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb.py) |  8   | ResNet50 | Kinetics400 |  65.95   |  91.05   |    5812    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb_20210630-cee5f725.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb/20210606_010153.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_hmdb51_rgb/20210606_010153.log.json) |

#### UCF101

| config                                  | gpus | backbone |  pretrain   | top1 acc | top5 acc | gpu_mem(M) |                 ckpt                  |                  log                  |                  json                  |
| :-------------------------------------- | :--: | :------: | :---------: | :------: | :------: | :--------: | :-----------------------------------: | :-----------------------------------: | :------------------------------------: |
| [slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb.py) |  8   | ResNet50 |  ImageNet   |  71.35   |  89.35   |    5812    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb_20210630-181e1661.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb/20210605_213503.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_ucf101_rgb/20210605_213503.log.json) |
| [slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb.py) |  8   | ResNet50 | Kinetics400 |  92.78   |  99.42   |    5812    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb_20210630-ee8c850f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb/20210606_010231.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_k400_pretrained_r50_8x4x1_40e_ucf101_rgb/20210606_010231.log.json) |

#### Something-Something V1

| config                                   | gpus | backbone | pretrain | top1 acc | top5 acc | gpu_mem(M) |                  ckpt                  |                  log                  |                  json                   |
| :--------------------------------------- | :--: | :------: | :------: | :------: | :------: | :--------: | :------------------------------------: | :-----------------------------------: | :-------------------------------------: |
| [slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb.py) |  8   | ResNet50 | ImageNet |  47.76   |  77.49   |    7759    | [ckpt](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb_20211202-d034ff12.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb/slowonly_imagenet_pretrained_r50_8x4x1_64e_sthv1_rgb.json) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowOnly model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    --work-dir work_dirs/slowonly_r50_4x16x1_256e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowOnly model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config              |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |       reference top1 acc        |        reference top5 acc        | inference_time(video/s) | gpu_mem(M) |        ckpt        |        log        |
| :------------------ | :------------: | :--: | :------: | :------: | :------: | :------: | :-----------------------------: | :------------------------------: | :---------------------: | :--------: | :----------------: | :---------------: |
| [tanet_r50_dense_1x1x8_100e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   |  TANet   | ImageNet |  76.22   |  92.31   | [76.22](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | [92.53](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) |            x            |    7125    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 8 GPUs x 8 videos/gpu and lr=0.04 for 16 GPUs x 16 videos/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. The checkpoints for reference repo can be downloaded [here](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing).
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TANet model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TANet model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                 |   resolution   | gpus |  backbone   |   pretrain   | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                 ckpt                 |                 log                  |
| :------------------------------------- | :------------: | :--: | :---------: | :----------: | :------: | :------: | :---------------------: | :--------: | :----------------------------------: | :----------------------------------: |
| [timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | TimeSformer | ImageNet-21K |  77.96   |  93.57   |            x            |   15235    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb_20220815-a4d0d01f.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb/20220614_113611.log) |
| [timesformer_jointST_8x32x1_15e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_jointST_8x32x1_15e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | TimeSformer | ImageNet-21K |  76.93   |  93.27   |            x            |   33358    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_jointST_8x32x1_15e_8xb8_kinetics400_rgb/timesformer_jointST_8x32x1_15e_8xb8_kinetics400_rgb_20220815-8022d1c0.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_jointST_8x32x1_15e_8xb8_kinetics400_rgb/20220614_180320.log) |
| [timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb.py) | short-side 320 |  8   | TimeSformer | ImageNet-21K |  76.98   |  92.83   |            x            |   12355    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb_20220815-78f05367.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb/20220615_101108.log) |

:::{note}

1. The **gpus** indicates the number of gpu (80G A100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.005 for 8 GPUs x 8 videos/gpu and lr=0.00375 for 8 GPUs x 6 videos/gpu.
2. We keep the test setting with the [original repo](https://github.com/facebookresearch/TimeSformer) (three crop x 1 clip).
3. The pretrained model `vit_base_patch16_224.pth` used by TimeSformer was converted from [vision_transformer](https://github.com/google-research/vision_transformer).

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TimeSformer model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TimeSformer model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                       | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) |            ckpt             |            log             |            json             |
| :--------------------------- | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------: | :-------------------------: | :------------------------: | :-------------------------: |
| [tin_r50_1x1x8_40e_sthv1_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py) | height 100 | 8x4  | ResNet50 | ImageNet |  44.25   |  73.94   |       44.04        |       72.72        |    6181    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/tin_r50_1x1x8_40e_sthv1_rgb_20200729-4a33db86.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log.json) |

#### Something-Something V2

| config                       | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M) |            ckpt             |            log             |            json             |
| :--------------------------- | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------: | :-------------------------: | :------------------------: | :-------------------------: |
| [tin_r50_1x1x8_40e_sthv2_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py) | height 240 | 8x4  | ResNet50 | ImageNet |  56.70   |  83.62   |       56.48        |       83.45        |    6185    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/tin_r50_1x1x8_40e_sthv2_rgb_20200912-b27a7337.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log.json) |

#### Kinetics-400

| config                              |   resolution   | gpus | backbone |    pretrain     | top1 acc | top5 acc | gpu_mem(M) |               ckpt                |               log                |               json                |
| :---------------------------------- | :------------: | :--: | :------: | :-------------: | :------: | :------: | :--------: | :-------------------------------: | :------------------------------: | :-------------------------------: |
| [tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb.py) | short-side 256 | 8x4  | ResNet50 | TSM-Kinetics400 |  70.89   |  89.89   |    6187    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb_20200810-4a146a70.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log.json) |

Here, we use `finetune` to indicate that we use [TSM model](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) trained on Kinetics-400 to finetune the TIN model on Kinetics-400.

:::{note}

1. The **reference topk acc** are got by training the [original repo ##1aacd0c](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) with no [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4).
   The [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4) will lead to incorrect performance, so we fix it before running.
2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
3. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
4. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
5. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TIN model on Something-Something V1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TIN model on Something-Something V1 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config           |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |      reference top1 acc      |      reference top5 acc       | inference_time(video/s) | gpu_mem(M) |      ckpt       |      log       |      json       |
| :--------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :--------------------------: | :---------------------------: | :---------------------: | :--------: | :-------------: | :------------: | :-------------: |
| [tpn_slowonly_r50_8x8x1_150e_kinetics_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py) | short-side 320 | 8x2  | ResNet50 |   None   |  73.58   |  91.35   |              x               |               x               |            x            |    6916    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb-c568e7ad.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.json) |
| [tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  76.59   |  92.72   | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) |            x            |    6916    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb-44362b55.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.json) |

#### Something-Something V1

| config                                | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | gpu_mem(M) |                 ckpt                 |                 log                 |                 json                 |
| :------------------------------------ | :--------: | :--: | :------: | :------: | :------: | :------: | :--------: | :----------------------------------: | :---------------------------------: | :----------------------------------: |
| [tpn_tsm_r50_1x1x8_150e_sthv1_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py) | height 100 | 8x6  | ResNet50 |   TSM    |  51.50   |  79.15   |    8828    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb_20211202-c28ed83f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.json) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TPN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb [--validate --seed 0 --deterministic]
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TPN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                              | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) | gpu_mem(M) |                ckpt                |                log                |
| :---------------------------------- | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :--------: | :--------------------------------: | :-------------------------------: |
| [trn_r50_1x1x8_50e_8xb16_sthv1_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv1_rgb.py) | height 100 |  8   | ResNet50 | ImageNet |         31.81 / 33.86         |         60.47 / 62.24         |   11037    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv1_rgb/trn_r50_1x1x8_50e_8xb16_sthv1_rgb_20220815-e13db2e9.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv1_rgb/20220808_143221.log) |

#### Something-Something V2

| config                              | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) | gpu_mem(M) |                ckpt                |                log                |
| :---------------------------------- | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :--------: | :--------------------------------: | :-------------------------------: |
| [trn_r50_1x1x8_50e_8xb16_sthv2_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv2_rgb.py) | height 240 |  8   | ResNet50 | ImageNet |         48.54 / 51.53         |         76.53 / 78.60         |   11037    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv2_rgb/trn_r50_1x1x8_50e_8xb16_sthv2_rgb_20220815-e01617db.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv2_rgb/20220808_143256.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip).
3. In the original [repository](https://github.com/zhoubolei/TRN-pytorch), the author augments data with random flipping on something-something dataset, but the augmentation method may be wrong due to the direct actions, such as `push left to right`. So, we replaced `flip` with `flip with label mapping`, and change the testing method `TenCrop`, which has five flipped crops, to `Twice Sample & ThreeCrop`.
4. We use `ResNet50` instead of `BNInception` as the backbone of TRN. When Training `TRN-ResNet50` on sthv1 dataset in the original repository, we get top1 (top5) accuracy 30.542 (58.627) vs. ours 31.81 (60.47).

:::

For more details on data preparation, you can refer to

- [preparing_sthv1](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv1/README.md)
- [preparing_sthv2](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv2/README.md)

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TRN model on sthv1 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv1_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TRN model on sthv1 dataset.

```shell
python tools/test.py configs/recognition/trn/trn_r50_1x1x8_50e_8xb16_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [tsm_r50_1x1x8_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  72.96   |  90.45   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_1x1x8_100e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_r50_1x1x8_100e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.11   |  90.06   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_1x1x16_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_r50_1x1x16_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.64   |  91.42   |            x            |   27044    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_dense_1x1x8_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_r50_dense_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.39   |  90.78   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_embedded_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.45   |  91.11   |            x            |   19726    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_dot_product_r50_1x1x8_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.17   |  90.95   |            x            |   18413    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.37   |  90.82   |            x            |   19925    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings. The checkpoints for reference repo can be downloaded [here](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_reference_ckpt.rar).
4. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip), which is referred from the [original repo](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd).
   We use efficient setting as default provided in config files, and it can be changed to accurate setting by

```python
...
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        ## `num_clips = 8` when using 8 segments
        num_clips=16,
        ## set `twice_sample=True` for twice sample in accurate setting
        twice_sample=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    ## dict(type='CenterCrop', crop_size=224), it is used for efficient setting
    dict(type='ThreeCrop', crop_size=256),  ## it is used for accurate setting
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
```

5. When applying Mixup and CutMix, we use the hyper parameter `alpha=0.2`.
6. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
7. The **infer_ckpt** means those checkpoints are ported from [TSM](https://github.com/mit-han-lab/temporal-shift-module/blob/master/test_models.py).

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](data_preparation.md).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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
@article{NonLocal2018,
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

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  72.77   |  90.66   |            x            |    8321    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_1x1x5_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_r50_1x1x5_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.73   |  91.15   |            x            |   13616    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_1x1x8_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_r50_1x1x8_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.21   |  91.36   |            x            |   21549    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_dense_1x1x5_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  71.37   |  89.66   |            x            |   13616    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |

#### Using backbones from 3rd-party in TSN

It's possible and convenient to use a 3rd-party backbone for TSN under the framework of MMAction2, here we provide some examples for:

- [x] Backbones from [MMClassification](https://github.com/open-mmlab/mmclassification/)
- [x] Backbones from [TorchVision](https://github.com/pytorch/vision/)
- [x] Backbones from [TIMM (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models)

| config                                   |   resolution   | gpus |                  backbone                  | pretrain | top1 acc | top5 acc |                  ckpt                  |                  log                   |
| :--------------------------------------- | :------------: | :--: | :----------------------------------------: | :------: | :------: | :------: | :------------------------------------: | :------------------------------------: |
| [tsn_rn101_32x4d_1x1x3_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_rn101_32x4d_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNeXt101-32x4d \[[MMCls](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)\] | ImageNet |  72.79   |  90.40   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb-16a8b561.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.log) |
| [tsn_dense161_1x1x3_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_dense161_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | Densenet-161 \[[TorchVision](https://github.com/pytorch/vision/)\] | ImageNet |  68.31   |  87.79   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb-cbe85332.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.log) |
| [tsn_swin_transformer_1x1x3_100e_8xb32_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn/custom_backbones/tsn_swin_transformer_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | Swin Transformer Base \[[timm](https://github.com/rwightman/pytorch-image-models)\] | ImageNet |  76.90   |  92.55   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb-805380f6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.log) |

1. Note that some backbones in TIMM are not supported due to multiple reasons. Please refer to to [PR ##880](https://github.com/open-mmlab/mmaction2/pull/880) for details.

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](https://github.com/open-mmlab/mmaction2/tree/master/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [preparing_ucf101](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/ucf101/README.md)
- [preparing_kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics/README.md)
- [preparing_sthv1](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv1/README.md)
- [preparing_sthv2](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/sthv2/README.md)
- [preparing_mit](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/mit/README.md)
- [preparing_mmit](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/mmit/README.md)
- [preparing_hvu](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/hvu/README.md)
- [preparing_hmdb51](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/hmdb51/README.md)

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb.py  \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                          |   resolution   | backbone | top1 10-view | top1 30-view |             reference top1 10-view              |             reference top1 30-view              |             ckpt              |
| :------------------------------ | :------------: | :------: | :----------: | :----------: | :---------------------------------------------: | :---------------------------------------------: | :---------------------------: |
| [x3d_s_13x6x1_facebook_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py) | short-side 320 |  X3D_S   |     72.7     |     73.3     | 73.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 73.5 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/x3d/facebook/x3d_s_13x6x1_facebook_kinetics400_rgb_20201027-623825a0.pth)\[1\] |
| [x3d_m_16x5x1_facebook_kinetics400_rgb](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py) | short-side 320 |  X3D_M   |     74.9     |     75.5     | 75.1 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | 76.2 \[[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)\] | [ckpt](https://download.openmmlab.com/mmaction/v2.0/recognition/x3d/facebook/x3d_m_16x5x1_facebook_kinetics400_rgb_20201027-3f42382a.pth)\[1\] |

\[1\] The models are ported from the repo [SlowFast](https://github.com/facebookresearch/SlowFast/) and tested on our data. Currently, we only support the testing of X3D models, training will be available soon.

:::{note}

1. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](data_preparation.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test X3D model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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
