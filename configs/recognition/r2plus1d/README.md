# R2plus1D

## Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{tran2018closer,
  title={A closer look at spatiotemporal convolutions for action recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={6450--6459},
  year={2018}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone | pretrain| top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py) | short-side 256|8x4| ResNet34|None |67.30|87.65|x|5019|[ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb_20200729-aa94765e.pth)|[log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/20200728_021421.log)|[json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_256p_8x8x1_180e_kinetics400_rgb/20200728_021421.log.json)|
|[r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py) | short-side 256|8| ResNet34|None |67.3|87.8|x|5019|[ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb_20200826-ab35a529.pth)|[log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/20200724_201360.log.json)|[json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb/20200724_201360.log)|
|[r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py) | short-side 320|8x2| ResNet34|None |68.68|88.36|1.6 (80x3 frames)|5019|[ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8x1_180e_kinetics400_rgb_20200618-3fce5629.pth)| [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r21d_8x8.log)| [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb/r2plus1d_r34_8x8_69.58_88.36.log.json)|
|[r2plus1d_r34_32x2x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb.py) |short-side 320|8x2| ResNet34|None |74.60|91.59|0.5 (320x3 frames)|12975| [ckpt](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2x1_180e_kinetics400_rgb_20200618-63462eb3.pth) | [log](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r21d_32x2.log)| [json](https://download.openmmlab.com/mmaction/recognition/r2plus1d/r2plus1d_r34_32x2x1_180e_kinetics400_rgb/r2plus1d_r34_32x2_74.6_91.6.log.json)|

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train R(2+1)D model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    --work-dir work_dirs/r2plus1d_r34_3d_8x8x1_180e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test R(2+1)D model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips=prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
