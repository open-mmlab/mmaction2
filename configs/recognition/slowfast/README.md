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

|config | resolution | gpus | backbone |pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py) |short-side 320|8| ResNet50|None |75.51|92.05|x|6331|[ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb_20200728-145f1097.pth)|[log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_256p_4x16x1_256e_kinetics400_rgb/20200731_151706.log)|
|[slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb.py) |short-side 320|8| ResNet50 |None|76.24|92.3|1.3 ((32+8)x10x3 frames)|9201| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/20200716_192653.log)|
|[slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb_steplr](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb_steplr.py) |short-side 320|8| ResNet50 |None|75.72|92.37|x|9401| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr-43988bac.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr.log)|
|[slowfast_r101_8x8x1_256e_8xb8_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r101_8x8x1_256e_8xb8_kinetics400_rgb.py) |short-side 320|8| ResNet101 |None|76.29|91.97|x|13454| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/20210218_121513.log)|
|[slowfast_r101_r50_4x16x1_256e_8xb8_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r101_r50_4x16x1_256e_8xb8_kinetics400_rgb.py) |short-side 320|8| ResNet101 + ResNet50 |None|76.33|93.03|x|8018| [ckpt](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/slowfast_r101_4x16x1_256e_kinetics400_rgb_20210218-d8b58813.pth) | [log](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_4x16x1_256e_kinetics400_rgb/20210118_133528.log)|

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

Example: train SlowFast model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test SlowFast model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/slowfast/slowfast_r50_4x16x1_256e_8xb8_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

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
