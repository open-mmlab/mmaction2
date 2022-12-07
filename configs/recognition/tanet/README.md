# TANet

[TAM: Temporal Adaptive Module for Video Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_TAM_Temporal_Adaptive_Module_for_Video_Recognition_ICCV_2021_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Video data is with complex temporal dynamics due to various factors such as camera motion, speed variation, and different activities. To effectively capture this diverse motion pattern, this paper presents a new temporal adaptive module ({\\bf TAM}) to generate video-specific temporal kernels based on its own feature map. TAM proposes a unique two-level adaptive modeling scheme by decoupling the dynamic kernel into a location sensitive importance map and a location invariant aggregation weight. The importance map is learned in a local temporal window to capture short-term information, while the aggregation weight is generated from a global view with a focus on long-term structure. TAM is a modular block and could be integrated into 2D CNNs to yield a powerful video architecture (TANet) with a very small extra computational cost. The extensive experiments on Kinetics-400 and Something-Something datasets demonstrate that our TAM outperforms other temporal modeling methods consistently, and achieves the state-of-the-art performance under the similar complexity.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018253-c3e1ba5b-ac35-4c55-be28-0134b76888e8.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| config                                                                                                                 |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc |                                              reference top1 acc                                              |                                              reference top5 acc                                              | inference_time(video/s) | gpu_mem(M) |                                                                                     ckpt                                                                                      |                                                                                 log                                                                                 |                                                                                 json                                                                                  |
| :--------------------------------------------------------------------------------------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---------------------: | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [tanet_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py) | short-side 320 |  8   |  TANet   | ImageNet |  76.28   |  92.60   | [76.22](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) | [92.53](https://github.com/liu-zhy/temporal-adaptive-module/blob/master/scripts/test_tam_kinetics_rgb_8f.sh) |            x            |    7124    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219.log) | [json](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb/tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219.json) |

### Something-Something V1

| config                                                                                         | resolution | gpus | backbone | pretrain | top1 acc (efficient/accurate) | top5 acc (efficient/accurate) | gpu_mem(M) |                                                                         ckpt                                                                          |                                                                log                                                                 |                                                                 json                                                                 |
| :--------------------------------------------------------------------------------------------- | :--------: | :--: | :------: | :------: | :---------------------------: | :---------------------------: | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: |
| [tanet_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb.py)   | height 100 |  8   |  TANet   | ImageNet |          47.34/49.58          |          75.72/77.31          |    7127    |  [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/tanet_r50_1x1x8_50e_sthv1_rgb_20210630-f4a48609.pth)  |         [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/20210606_205006.log)         |       [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x8_50e_sthv1_rgb/20210606_205006.log.json)       |
| [tanet_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb.py) | height 100 |  8   |  TANet   | ImageNet |          49.05/50.91          |          77.90/79.13          |    7127    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb_20211202-370c2128.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb.log) | [ckpt](https://download.openmmlab.com/mmaction/recognition/tanet/tanet_r50_1x1x16_50e_sthv1_rgb/tanet_r50_1x1x16_50e_sthv1_rgb.json) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 8 GPUs x 8 videos/gpu and lr=0.04 for 16 GPUs x 16 videos/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time, not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. The checkpoints for reference repo can be downloaded [here](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing).
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](/docs/en/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TANet model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tanet_r50_dense_1x1x8_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/en/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TANet model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tanet/tanet_r50_dense_1x1x8_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/en/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@article{liu2020tam,
  title={TAM: Temporal Adaptive Module for Video Recognition},
  author={Liu, Zhaoyang and Wang, Limin and Wu, Wayne and Qian, Chen and Lu, Tong},
  journal={arXiv preprint arXiv:2005.06803},
  year={2020}
}
```
