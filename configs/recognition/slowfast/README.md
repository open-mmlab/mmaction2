# SlowFast

[SlowFast Networks for Video Recognition](https://openaccess.thecvf.com/content_ICCV_2019/html/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present SlowFast networks for video recognition. Our model involves (i) a Slow pathway, operating at low frame rate, to capture spatial semantics, and (ii) a Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution. The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition. Our models achieve strong performance for both action classification and detection in video, and large improvements are pin-pointed as contributions by our SlowFast concept. We report state-of-the-art accuracy on major video recognition benchmarks, Kinetics, Charades and AVA.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143044111-94676f64-7ba8-4081-9011-f8054bed7030.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy |    scheduler     | resolution | gpus |       backbone       | pretrain | top1 acc | top5 acc | testing protocol  | FLOPs | params |           config           |           ckpt            |           log            |
| :---------------------: | :--------------: | :--------: | :--: | :------------------: | :------: | :------: | :------: | :---------------: | :---: | :----: | :------------------------: | :-----------------------: | :----------------------: |
|         4x16x1          |  Linear+Cosine   |  224x224   |  8   |       ResNet50       |   None   |  75.55   |  92.35   | 10 clips x 3 crop | 36.3G | 34.5M  | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |       ResNet50       |   None   |  76.80   |  92.99   | 10 clips x 3 crop | 66.1G | 34.6M  | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|          8x8x1          | Linear+MultiStep |  224x224   |  8   |       ResNet50       |   None   |  76.65   |  92.86   | 10 clips x 3 crop | 66.1G | 34.6M  | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb_20220818-b62a501f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   |  224x224   |  8   |      ResNet101       |   None   |  78.65   |  93.88   | 10 clips x 3 crop | 126G  | 62.9M  | [config](/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|         4x16x1          |  Linear+Cosine   |  224x224   |  32  | ResNet101 + ResNet50 |   None   |  77.03   |  92.99   | 10 clips x 3 crop | 64.9G | 62.4M  | [config](/configs/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb_20220901-a77ac3ee.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{feichtenhofer2019slowfast,
  title={Slowfast networks for video recognition},
  author={Feichtenhofer, Christoph and Fan, Haoqi and Malik, Jitendra and He, Kaiming},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={6202--6211},
  year={2019}
}
```
