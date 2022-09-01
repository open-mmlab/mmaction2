# CSN

[Video Classification With Channel-Separated Convolutional Networks](https://openaccess.thecvf.com/content_ICCV_2019/html/Tran_Video_Classification_With_Channel-Separated_Convolutional_Networks_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Group convolution has been shown to offer great computational savings in various 2D convolutional architectures for image classification. It is natural to ask: 1) if group convolution can help to alleviate the high computational cost of video classification networks; 2) what factors matter the most in 3D group convolutional networks; and 3) what are good computation/accuracy trade-offs with 3D group convolutional networks. This paper studies the effects of different design choices in 3D group convolutional networks for video classification. We empirically demonstrate that the amount of channel interactions plays an important role in the accuracy of 3D group convolutional networks. Our experiments suggest two main findings. First, it is a good practice to factorize 3D convolutions by separating channel interactions and spatiotemporal interactions as this leads to improved accuracy and lower computational cost. Second, 3D channel-separated convolutions provide a form of regularization, yielding lower training accuracy but higher test accuracy compared to 3D convolutions. These two empirical findings lead us to design an architecture -- Channel-Separated Convolutional Network (CSN) -- which is simple, efficient, yet accurate. On Sports1M, Kinetics, and Something-Something, our CSNs are comparable with or better than the state-of-the-art while being 2-3 times more efficient.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143017317-1bd7e557-7d99-4964-8b89-ab5280945d54.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy |   resolution   | gpus |        backbone         | pretrain | top1 acc | top5 acc | testing protocol  | inference time(video/s) | gpu_mem(M) |         config         |         ckpt         |         log          |
| :---------------------: | :------------: | :--: | :---------------------: | :------: | :------: | :------: | :---------------: | :---------------------: | :--------: | :--------------------: | :------------------: | :------------------: |
|         32x2x1          | short-side 320 |  8   |     ResNet152 (IR)      |  IG65M   |  82.66   |  95.82   | 10 clips x 3 crop |            x            |   32703    | [config](/configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb_20220811-c7a3cc5b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          | short-side 320 |  8   | ResNet152 (IR+BNFrozen) |  IG65M   |  82.58   |  95.76   | 10 clips x 3 crop |            x            |   32703    | [config](/configs/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-7d1dacde.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          | short-side 320 |  8   | ResNet50 (IR+BNFrozen)  |  IG65M   |  79.17   |  94.14   | 10 clips x 3 crop |            x            |   22238    | [config](/configs/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb_20220811-44395bae.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/csn/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb/ircsn_ig65m-pretrained-r50-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.log) |
|         32x2x1          | short-side 320 |  x   |     ResNet152 (IP)      |   None   |  77.69   |  92.83   | 10 clips x 3 crop |            x            |     x      | [config](/configs/recognition/csn/ipcsn_r152_32x2x1-180e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-d565828d.pth) |          x           |
|         32x2x1          | short-side 320 |  x   |     ResNet152 (IR)      |   None   |  79.17   |  94.14   | 10 clips x 3 crop |            x            |     x      | [config](/configs/recognition/csn/ircsn_r152_32x2x1-180e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-5c933ae1.pth) |          x           |
|         32x2x1          | short-side 320 |  x   | ResNet152 (IP+BNFrozen) |  IG65M   |  82.51   |  95.52   | 10 clips x 3 crop |            x            |     x      | [config](/configs/recognition/csn/ipcsn_ig65m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth) |          x           |
|         32x2x1          | short-side 320 |  x   | ResNet152 (IP+BNFrozen) | Sports1M |  78.77   |  93.78   | 10 clips x 3 crop |            x            |     x      | [config](/configs/recognition/csn/ipcsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth) |          x           |
|         32x2x1          | short-side 320 |  x   | ResNet152 (IR+BNFrozen) | Sports1M |  78.82   |  93.34   | 10 clips x 3 crop |            x            |     x      | [config](/configs/recognition/csn/ircsn_sports1m-pretrained-r152-bnfrozen_32x2x1-58e_kinetics400-rgb.py) | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-b9b10241.pth) |          x           |

1. The **gpus** indicates the number of gpu (80G A100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
3. The **infer_ckpt** means those checkpoints are ported from [VMZ](https://github.com/facebookresearch/VMZ).

For more details on data preparation, you can refer to the **Prepare videos** part in the [Data Preparation Tutorial](/docs/en/user_guides/2_data_prepare.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CSN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py  \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CSN model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/csn/ircsn_ig65m-pretrained-r152_8xb12-32x2x1-58e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

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
