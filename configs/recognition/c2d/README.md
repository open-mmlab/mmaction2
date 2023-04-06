# C2D

<!-- [ALGORITHM] -->

[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

## Abstract

<!-- [ABSTRACT] -->

Both convolutional and recurrent operations are building blocks that process one local neighborhood at a time. In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions. This building block can be plugged into many computer vision architectures. On the task of video classification, even without any bells and whistles, our non-local models can compete or outperform current competition winners on both Kinetics and Charades datasets. In static image recognition, our non-local models improve object detection/segmentation and pose estimation on the COCO suite of tasks.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/37479394/195281946-b70c76fd-d424-4371-95cf-09f51f20eda0.jpg" width="800"/>

</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | scheduler | resolution | gpus |    backbone     | pretrain | top1 acc | top5 acc |    reference top1 acc     |    reference top5 acc     | testing protocol  | FLOPs | params |     config     |     ckpt     |     log     |
| :---------------------: | :-------: | :--------: | :--: | :-------------: | :------: | :------: | :------: | :-----------------------: | :-----------------------: | :---------------: | :---: | :----: | :------------: | :----------: | :---------: |
|          8x8x1          | MultiStep |  224x224   |  8   |  ResNet50<br>   | ImageNet |  73.44   |  91.00   | 67.2<br>[\[PySlowFast\]](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | 87.8<br>[\[PySlowFast\]](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | 10 clips x 3 crop |  33G  | 24.3M  | [config](/configs/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb_20221027-e0227b22.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|          8x8x1          | MultiStep |  224x224   |  8   |  ResNet101<br>  | ImageNet |  74.97   |  91.77   |             x             |             x             | 10 clips x 3 crop |  63G  | 43.3M  | [config](/configs/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb_20221027-557bd8bc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r101-in1k-pre-nopool_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|          8x8x1          | MultiStep |  224x224   |  8   | ResNet50<br>(TemporalPool) | ImageNet |  73.89   |  91.21   | 71.9<br>[\[Non-Local\]](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | 90.0<br>[\[Non-Local\]](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | 10 clips x 3 crop |  19G  | 24.3M  | [config](/configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb_20221027-3ca304fa.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.log) |
|         16x4x1          | MultiStep |  224x224   |  8   | ResNet50<br>(TemporalPool) | ImageNet |  74.97   |  91.91   |             x             |             x             | 10 clips x 3 crop |  39G  | 24.3M  | [config](/configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb_20221027-5f382a43.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/c2d/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb/c2d_r50-in1k-pre_8xb32-16x4x1-100e_kinetics400-rgb.log) |

1. The values in columns named after "reference" are the results reported in the original repo, using the same model settings.
2. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train C2D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py  \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test C2D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/c2d/c2d_r50-in1k-pre_8xb32-8x8x1-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{XiaolongWang2017NonlocalNN,
  title={Non-local Neural Networks},
  author={Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2017}
}
```
