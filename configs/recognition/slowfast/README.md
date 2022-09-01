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

| frame sampling strategy |    scheduler     |   resolution   | gpus |       backbone       | pretrain | top1 acc | top5 acc | testing protocol  | inference time(video/s) | gpu_mem(M) |       config       |       ckpt       |       log       |
| :---------------------: | :--------------: | :------------: | :--: | :------------------: | :------: | :------: | :------: | :---------------: | :---------------------: | :--------: | :----------------: | :--------------: | :-------------: |
|         4x16x1          |  Linear+Cosine   | short-side 320 |  8   |       ResNet50       |   None   |  75.27   |  92.27   | 10 clips x 3 crop |            x            |    6332    | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   | short-side 320 |  8   |       ResNet50       |   None   |  76.31   |  92.88   | 10 clips x 3 crop |            x            |    9201    | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|          8x8x1          | Linear+MultiStep | short-side 320 |  8   |       ResNet50       |   None   |  76.33   |  92.66   | 10 clips x 3 crop |            x            |    9395    | [config](/configs/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb_20220818-b62a501f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb/slowfast_r50_8xb8-8x8x1-steplr-256e_kinetics400-rgb.log) |
|          8x8x1          |  Linear+Cosine   | short-side 320 |  8   |      ResNet101       |   None   |  78.30   |  93.77   | 10 clips x 3 crop |            x            |   13431    | [config](/configs/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb_20220818-9c0e09bd.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb/slowfast_r101_8xb8-8x8x1-256e_kinetics400-rgb.log) |
|         4x16x1          |  Linear+Cosine   | short-side 320 |  32  | ResNet101 + ResNet50 |   None   |  76.68   |  92.82   | 10 clips x 3 crop |            x            |    8039    | [config](/configs/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb_20220901-a77ac3ee.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb/slowfast_r101-r50_32xb8-4x16x1-256e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to the **Prepare videos** part in the [Data Preparation Tutorial](/docs/en/user_guides/2_data_prepare.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

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

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

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
