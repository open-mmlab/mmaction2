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

|config | resolution | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb](/configs/recognition/csn/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb.py)|short-side 320|x| ResNet50 | None | 73.6 | 91.3 | x | x | [ckpt](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb_20210618-4e29e2e8.pth) | [log](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb/20210618_182414.log) | [json](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_bnfrozen_r50_32x2x1_180e_kinetics400_rgb/20210618_182414.log.json) |
|[ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb](/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r50_32x2x1_58e_kinetics400_rgb.py)|short-side 320|x| ResNet50 | IG65M | 79.0 | 94.2 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_ig65m_pretrained_r50_32x2x1_58e_kinetics400_rgb_20210617-86d33018.pth) | x | x |
|[ircsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb](/configs/recognition/csn/ircsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb.py)|short-side 320|x| ResNet152 | None | 76.5 | 92.1 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-5c933ae1.pth) | x | x |
|[ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](/configs/recognition/csn/ircsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py)|short-side 320|x| ResNet152 | Sports1M | 78.2 | 93.0 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-b9b10241.pth) | x | x |
|[ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py](/configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py)|short-side 320|8x4| ResNet152 | IG65M|82.76/82.6|95.68/95.3|x|8516|[ckpt](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth)/[infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-e63ee1bd.pth)|[log](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/20200809_053132.log)|[json](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb/20200809_053132.log.json)|
|[ipcsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb](/configs/recognition/csn/ipcsn_bnfrozen_r152_32x2x1_180e_kinetics400_rgb.py)|short-side 320|x| ResNet152 | None | 77.8 | 92.8 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_from_scratch_r152_32x2x1_180e_kinetics400_rgb_20210617-d565828d.pth) | x | x |
|[ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](/configs/recognition/csn/ipcsn_sports1m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py)|short-side 320|x| ResNet152 | Sports1M | 78.8 | 93.5 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_sports1m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-3367437a.pth) | x | x |
|[ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb](/configs/recognition/csn/ipcsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py)|short-side 320|x| ResNet152 | IG65M | 82.5 | 95.3 | x | x | [infer_ckpt](https://download.openmmlab.com/mmaction/recognition/csn/vmz/vmz_ipcsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20210617-c3be9793.pth) | x | x |
|[ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py](/configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py)|short-side 320|8x4| ResNet152 | IG65M|80.14|94.93|x|8517|[ckpt](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb_20200803-fc66ce8d.pth)|[log](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb/20200728_031952.log)|[json](https://download.openmmlab.com/mmaction/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb/20200728_031952.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu (32G V100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. The **infer_ckpt** means those checkpoints are ported from [VMZ](https://github.com/facebookresearch/VMZ).

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train CSN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py \
    --work-dir work_dirs/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test CSN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/csn/ircsn_ig65m_pretrained_r152_32x2x1_58e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

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
