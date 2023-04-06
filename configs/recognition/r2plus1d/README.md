# R2plus1D

[A closer look at spatiotemporal convolutions for action recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In this paper we discuss several forms of spatiotemporal convolutions for video analysis and study their effects on action recognition. Our motivation stems from the observation that 2D CNNs applied to individual frames of the video have remained solid performers in action recognition. In this work we empirically demonstrate the accuracy advantages of 3D CNNs over 2D CNNs within the framework of residual learning. Furthermore, we show that factorizing the 3D convolutional filters into separate spatial and temporal components yields significantly advantages in accuracy. Our empirical study leads to the design of a new spatiotemporal convolutional block "R(2+1)D" which gives rise to CNNs that achieve results comparable or superior to the state-of-the-art on Sports-1M, Kinetics, UCF101 and HMDB51.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143043885-3d00413c-b556-445e-9673-f5805c08c195.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |                config                |                ckpt                |                log                |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :----------------------------------: | :--------------------------------: | :-------------------------------: |
|          8x8x1          |  224x224   |  8   | ResNet34 |   None   |  69.76   |  88.41   | 10 clips x 3 crop | 53.1G | 63.8M  | [config](/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb_20220812-47cfe041.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.log) |
|         32x2x1          |  224x224   |  8   | ResNet34 |   None   |  75.46   |  92.28   | 10 clips x 3 crop | 213G  | 63.8M  | [config](/configs/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb_20220812-4270588c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/r2plus1d/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb/r2plus1d_r34_8xb8-32x2x1-180e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8xb8-8x8x1-180e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{tran2018closer,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={6450--6459},
  year={2018}
}
```
