# C2D

<!-- [ALGORITHM] -->

C2D is the baseline of [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

## Abstract

<!-- [ABSTRACT] -->

Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our non-local models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks. Code is available at [this https URL](https://github.com/facebookresearch/video-nonlocal-net) .

<!-- [IMAGE] -->

<div align=center>
<img src="TODO" width="300"/>
Baseline ResNet-50 C2D model. The dimensions of 3D output maps and filter kernels are in T×H×W (2D kernels in H×W), with the number of channels following. The input is 8×224×224. Residual blocks are shown in brackets.
</div>

NOTICE:

- C2D implementations are slightly different between 1.The paper above; 2."SlowFast" repo; 3."Video-Nonlocal-Net" repo.
- C2D implementation in MMAction2 is kept same as the ["Video-Nonlocal-Net" repo](https://github.com/facebookresearch/video-nonlocal-net/tree/main/scripts/run_c2d_baseline_400k.sh)
- Specifically:
  - pool_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
  - pool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
- C2D_Nopool implementation in MMAction2 is kept same as the ["SlowFast" repo](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml)

## Results and Models

### Kinetics-400

| frame sampling strategy | scheduler |  resolution   | gpus |  backbone   | pretrain | top1 acc | top5 acc |  reference top1 acc   |  reference top5 acc   | testing protocol  | inference time(video/s) | gpu_mem(M) |  config   |   ckpt   |   log   |
| :---------------------: | :-------: | :-----------: | :--: | :---------: | :------: | :------: | :------: | :-------------------: | :-------------------: | :---------------: | :---------------------: | :--------: | :-------: | :------: | :-----: |
|          8x8x1          | MultiStep | short-side 320 |  8   | ResNet50<br>(original) | ImageNet |  73.16   |  90.88   | [67.2](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | [87.8](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | 10 clips x 3 crop |            x            |   21547    | [config](/configs/recognition/c2d/c2d_nopool_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|          8x8x1          | MultiStep | short-side 320 |  8   | ResNet101<br>(original) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_nopool_imagenet-pretrained-r101_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|          8x8x1          | MultiStep | short-side 320 |  8   | ResNet50<br>(temporal pooling) | ImageNet |  71.78   |  89.90   | [71.9](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | [90.0](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | 10 clips x 3 crop |            x            |   17006    | [config](/configs/recognition/c2d/c2d_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|          8x8x1          | MultiStep | short-side 320 |  8   | ResNet101<br>(temporal pooling) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_imagenet-pretrained-r101_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|         16x4x1          | MultiStep | short-side 320 |  8   | ResNet50<br>(original) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_nopool_imagenet-pretrained-r50_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|         16x4x1          | MultiStep | short-side 320 |  8   | ResNet101<br>(original) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_nopool_imagenet-pretrained-r101_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|         16x4x1          | MultiStep | short-side 320 |  8   | ResNet50<br>(temporal pooling) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_imagenet-pretrained-r50_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|         16x4x1          | MultiStep | short-side 320 |  8   | ResNet101<br>(temporal pooling) | ImageNet |   TODO   |   TODO   |           x           |           x           | 10 clips x 3 crop |            x            |    TODO    | [config](/configs/recognition/c2d/c2d_imagenet-pretrained-r101_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |

1. The values in columns named after "reference" are the results reported in the original repo, using the same model settings.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [preparing_kinetics](/tools/data/kinetics/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C2D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/c2d/c2d_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py  \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C2D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/c2d/c2d_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

```BibTeX
@article{XiaolongWang2017NonlocalNN,
  title={Non-local Neural Networks},
  author={Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2017}
}
```
