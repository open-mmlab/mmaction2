# I3D

[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)

[Non-local Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The paucity of videos in current action classification datasets (UCF-101 and HMDB-51) has made it difficult to identify good video architectures, as most methods obtain similar performance on existing small-scale benchmarks. This paper re-evaluates state-of-the-art architectures in light of the new Kinetics Human Action Video dataset. Kinetics has two orders of magnitude more data, with 400 human action classes and over 400 clips per class, and is collected from realistic, challenging YouTube videos. We provide an analysis on how current architectures fare on the task of action classification on this dataset and how much performance improves on the smaller benchmark datasets after pre-training on Kinetics. We also introduce a new Two-Stream Inflated 3D ConvNet (I3D) that is based on 2D ConvNet inflation: filters and pooling kernels of very deep image classification ConvNets are expanded into 3D, making it possible to learn seamless spatio-temporal feature extractors from video while leveraging successful ImageNet architecture designs and even their parameters. We show that, after pre-training on Kinetics, I3D models considerably improve upon the state-of-the-art in action classification, reaching 80.9% on HMDB-51 and 98.0% on UCF-101.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043624-1944704a-5d3e-4a3f-b258-1505c49f6092.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus |           backbone            | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs  | params |            config            |            ckpt             |            log             |
| :---------------------: | :--------: | :--: | :---------------------------: | :------: | :------: | :------: | :---------------: | :----: | :----: | :--------------------------: | :-------------------------: | :------------------------: |
|         32x2x1          |  224x224   |  8   | ResNet50 (NonLocalDotProduct) | ImageNet |  74.80   |  92.07   | 10 clips x 3 crop | 59.3G  | 35.4M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb_20220812-8e1f2148.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-dot-product_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet50 (NonLocalEmbedGauss) | ImageNet |  74.73   |  91.80   | 10 clips x 3 crop | 59.3G  | 35.4M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb_20220812-afd8f562.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-embedded-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |   ResNet50 (NonLocalGauss)    | ImageNet |  73.97   |  91.33   | 10 clips x 3 crop |  56.5  | 31.7M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb_20220812-0c5cbf5a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-nl-gaussian_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |           ResNet50            | ImageNet |  73.47   |  91.27   | 10 clips x 3 crop | 43.5G  | 28.0M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.log) |
|      dense-32x2x1       |  224x224   |  8   |           ResNet50            | ImageNet |  73.77   |  91.35   | 10 clips x 3 crop | 43.5G  | 28.0M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb_20220812-9f46003f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50_8xb8-dense-32x2x1-100e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   |       ResNet50 (Heavy)        | ImageNet |  76.21   |  92.48   | 10 clips x 3 crop | 166.3G | 33.0M  | [config](/configs/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb_20220812-ed501b31.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/i3d/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb/i3d_imagenet-pretrained-r50-heavy_8xb8-32x2x1-100e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train I3D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/i3d/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

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
