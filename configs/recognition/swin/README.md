# VideoSwin

[Video Swin Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The vision community is witnessing a modeling shift from CNNs to Transformers, where pure Transformer architectures have attained top accuracy on the major video recognition benchmarks. These video models are all built on Transformer layers that globally connect patches across the spatial and temporal dimensions. In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including on action recognition (84.9 top-1 accuracy on Kinetics-400 and 85.9 top-1 accuracy on Kinetics-600 with ~20xless pre-training data and ~3xsmaller model size) and temporal modeling (69.6 top-1 accuracy on Something-Something v2).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/58767402/191190475-3aecf940-c254-47fa-96a7-df2d2b3bae68.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone |   pretrain   | top1 acc | top5 acc |      reference top1 acc      |      reference top1 acc      | testing protocol | FLOPs | params |      config      |      ckpt      |      log       |
| :---------------------: | :--------: | :--: | :------: | :----------: | :------: | :------: | :--------------------------: | :--------------------------: | :--------------: | :---: | :----: | :--------------: | :------------: | :------------: |
|         32x2x1          |  224x224   |  8   |  Swin-T  | ImageNet-1k  |  78.90   |  93.77   | 78.84 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 93.76 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop |  88G  | 28.2M  | [config](/configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-S  | ImageNet-1k  |  80.54   |  94.46   | 80.58 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.45 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop | 166G  | 49.8M  | [config](/configs/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-B  | ImageNet-1k  |  80.57   |  94.49   | 80.55 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 94.66 \[[VideoSwin](https://github.com/SwinTransformer/Video-Swin-Transformer)\] | 4 clips x 3 crop | 282G  | 88.0M  | [config](/configs/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |  Swin-L  | ImageNet-22k |  83.46   |  95.91   |            83.1\*            |            95.9\*            | 4 clips x 3 crop | 604G  |  197M  | [config](/configs/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-78ad8b11.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-large-p244-w877_in22k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.log) |

### Kinetics-700

| frame sampling strategy | resolution | gpus | backbone |   pretrain   | top1 acc | top5 acc | testing protocol | FLOPs | params |               config                |               ckpt                |               log                |
| :---------------------: | :--------: | :--: | :------: | :----------: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------: | :-------------------------------: | :------------------------------: |
|         32x2x1          |  224x224   |  16  |  Swin-L  | ImageNet-22k |  75.92   |  92.72   | 4 clips x 3 crop | 604G  |  197M  | [config](/configs/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.log) |

### Kinetics-710

| frame sampling strategy | resolution | gpus | backbone |  pretrain   | top1 acc | top5 acc | testing protocol | FLOPs | params |               config                |               ckpt                |                log                |
| :---------------------: | :--------: | :--: | :------: | :---------: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------: | :-------------------------------: | :-------------------------------: |
|         32x2x1          |  224x224   |  32  |  Swin-S  | ImageNet-1k |  76.90   |  92.96   | 4 clips x 3 crop | 604G  |  197M  | [config](/configs/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb_20230612-8e082ff1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb/swin-small-p244-w877_in1k-pre_32xb4-amp-32x2x1-30e_kinetics710-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. `*` means that the numbers are copied from the paper.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
4. Pre-trained image models can be downloaded from [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer#main-results-on-imagenet-with-pretrained-models).

For more details on data preparation, you can refer to [Kinetics](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train VideoSwin model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test VideoSwin model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{liu2022video,
  title={Video swin transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3202--3211},
  year={2022}
}
```
