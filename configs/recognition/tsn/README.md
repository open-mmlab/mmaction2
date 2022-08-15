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

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  72.77   |  90.66   |            x            |    8321    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_1x1x5_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x5_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.73   |  91.15   |            x            |   13616    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_1x1x8_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x8_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.21   |  91.36   |            x            |   21549    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |
| [tsn_r50_dense_1x1x5_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_dense_1x1x5_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  71.37   |  89.66   |            x            |   13616    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb/20200614_063526.log) |

### Using backbones from 3rd-party in TSN

It's possible and convenient to use a 3rd-party backbone for TSN under the framework of MMAction2, here we provide some examples for:

- [x] Backbones from [MMClassification](https://github.com/open-mmlab/mmclassification/)
- [x] Backbones from [TorchVision](https://github.com/pytorch/vision/)
- [x] Backbones from [TIMM (pytorch-image-models)](https://github.com/rwightman/pytorch-image-models)

| config                                   |   resolution   | gpus |                  backbone                  | pretrain | top1 acc | top5 acc |                  ckpt                  |                  log                   |
| :--------------------------------------- | :------------: | :--: | :----------------------------------------: | :------: | :------: | :------: | :------------------------------------: | :------------------------------------: |
| [tsn_rn101_32x4d_1x1x3_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_rn101_32x4d_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | ResNeXt101-32x4d \[[MMCls](https://github.com/open-mmlab/mmclassification/tree/master/configs/resnext)\] | ImageNet |  72.79   |  90.40   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb-16a8b561.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_rn101_32x4d_320p_1x1x3_100e_kinetics400_rgb.log) |
| [tsn_dense161_1x1x3_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_dense161_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | Densenet-161 \[[TorchVision](https://github.com/pytorch/vision/)\] | ImageNet |  68.31   |  87.79   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb-cbe85332.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb/tsn_dense161_320p_1x1x3_100e_kinetics400_rgb.log) |
| [tsn_swin_transformer_1x1x3_100e_8xb32_kinetics400_rgb](/configs/recognition/tsn/custom_backbones/tsn_swin_transformer_1x1x3_100e_8xb32_kinetics400_rgb.py) | short-side 320 |  8   | Swin Transformer Base \[[timm](https://github.com/rwightman/pytorch-image-models)\] | ImageNet |  76.90   |  92.55   | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb-805380f6.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsn/custom_backbones/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb/tsn_swin_transformer_video_320p_1x1x3_100e_kinetics400_rgb.log) |

1. Note that some backbones in TIMM are not supported due to multiple reasons. Please refer to to [PR #880](https://github.com/open-mmlab/mmaction2/pull/880) for details.

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [preparing_ucf101](/tools/data/ucf101/README.md)
- [preparing_kinetics](/tools/data/kinetics/README.md)
- [preparing_sthv1](/tools/data/sthv1/README.md)
- [preparing_sthv2](/tools/data/sthv2/README.md)
- [preparing_mit](/tools/data/mit/README.md)
- [preparing_mmit](/tools/data/mmit/README.md)
- [preparing_hvu](/tools/data/hvu/README.md)
- [preparing_hmdb51](/tools/data/hmdb51/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

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
