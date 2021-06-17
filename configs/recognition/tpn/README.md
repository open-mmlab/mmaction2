# TPN

## Introduction

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{yang2020tpn,
  title={Temporal Pyramid Network for Action Recognition},
  author={Yang, Ceyuan and Xu, Yinghao and Shi, Jianping and Dai, Bo and Zhou, Bolei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone | pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|short-side 320|8x4| ResNet50 | ImageNet | 73.10 | 91.03 | x | x | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb_20200910-b796d7a0.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/20200910_134330.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb/20200910_134330.log.json) |
|[tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb](/configs/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb.py)|short-side 320|8x4| ResNet50 | ImageNet | 76.20 | 92.44 | [75.49](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | [92.05](https://github.com/decisionforce/TPN/blob/master/MODELZOO.md) | x | 6916 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb_20200923-52629684.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/20200923_151919.log) | [json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_imagenet_pretrained_slowonly_r50_8x8x1_150e_kinetics_rgb/20200923_151919.log.json) |

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc| top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tpn_tsm_r50_1x1x8_150e_sthv1_rgb](/configs/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb.py)|height 100|8x6| ResNet50 | TSM | 50.80 | 79.05 | 8828 |[ckpt](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/tpn_tsm_r50_1x1x8_150e_sthv1_rgb_20210311-28de4cd5.pth) |[log](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/20210311_162636.log)|[json](https://download.openmmlab.com/mmaction/recognition/tpn/tpn_tsm_r50_1x1x8_150e_sthv1_rgb/20210311_162636.log.json)|

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by testing the checkpoint released on the original repo and codes, using the same dataset with ours.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TPN model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    --work-dir work_dirs/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb [--validate --seed 0 --deterministic]
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TPN model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tpn/tpn_slowonly_r50_8x8x1_150e_kinetics_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
