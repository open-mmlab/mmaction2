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

|config | resolution | gpus | backbone | pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py) | short-side 320|8| ResNet34|None |69.35|88.32|x|5036|[ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb_20200729-aa94765e.pth)|[log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/20200728_021421.log)|
|[r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb.py](/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_8xb8_kinetics400_rgb.py) | short-side 320|8| ResNet34|None |75.27|92.03|x|17006|[ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb_20200826-ab35a529.pth)|[log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/20200724_201360.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

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
