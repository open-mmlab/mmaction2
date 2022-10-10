# C2D

<!-- [ALGORITHM] -->
C2D is the baseline of [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)

NOTICE: C2D implementations are slightly different between 1.The paper above; 2."SlowFast" repo; 3."Video-Nonlocal-Net" repo.

C2D implementation in MMAction2 is kept same as the ["Video-Nonlocal-Net" repo](https://github.com/facebookresearch/video-nonlocal-net/tree/main/scripts/run_c2d_baseline_400k.sh)

Specifically:
- maxpool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
- maxpool3d_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))


C2D_Nopool implementation in MMAction2 is kept same as the ["SlowFast" repo](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml)


<!-- [ABSTRACT] -->


<!-- [IMAGE] -->


## Results and Models

### Kinetics-400

| frame sampling strategy | scheduler |   resolution   | gpus  |             backbone             | pretrain | top1 acc | top5 acc |                                           ref top1 acc                                           |                                           ref top5 acc                                           | testing protocol  | inference time(video/s) | gpu_mem(M) |                                                  config                                                   |       ckpt        |       log        |
| :---------------------: | :-------: | :------------: | :---: | :------------------------------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :---------------: | :---------------------: | :--------: | :-------------------------------------------------------------------------------------------------------: | :---------------: | :--------------: |
|          8x8x1          | MultiStep | short-side 320 |   8   |       ResNet50 (original)        | ImageNet |  73.16   |  90.88   | [67.2](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | [87.8](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | 10 clips x 3 crop |            x            |   21547    | [config](/configs/recognition/c2d/c2d_nopool_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py) | [ckpt TODO](TODO) | [log TODO](TODO) |
|          8x8x1          | MultiStep | short-side 320 |   8   | ResNet50 (with temporal pooling) | ImageNet |   TODO   |   TODO   | [71.9](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | [90.0](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | 10 clips x 3 crop |            x            |    TODO    |    [config](/configs/recognition/c2d/c2d_imagenet-pretrained-r50_8xb32-8x8x1-100e_kinetics400-rgb.py)     | [ckpt TODO](TODO) | [log TODO](TODO) |


1. The values in columns named after "reference" are the results reported in the original repo, using the same model settings.

2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.

3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.




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

Example: train TSN model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py  \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSN model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py \
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
