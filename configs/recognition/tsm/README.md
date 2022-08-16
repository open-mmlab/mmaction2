# TSM

[TSM: Temporal Shift Module for Efficient Video Understanding](https://openaccess.thecvf.com/content_ICCV_2019/html/Lin_TSM_Temporal_Shift_Module_for_Efficient_Video_Understanding_ICCV_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

The explosive growth in video streaming gives rise to challenges on performing video understanding at high accuracy and low computation cost. Conventional 2D CNNs are computationally cheap but cannot capture temporal relationships; 3D CNN based methods can achieve good performance but are computationally intensive, making it expensive to deploy. In this paper, we propose a generic and effective Temporal Shift Module (TSM) that enjoys both high efficiency and high performance. Specifically, it can achieve the performance of 3D CNN but maintain 2D CNN's complexity. TSM shifts part of the channels along the temporal dimension; thus facilitate information exchanged among neighboring frames. It can be inserted into 2D CNNs to achieve temporal modeling at zero computation and zero parameters. We also extended TSM to online setting, which enables real-time low-latency online video recognition and video object detection. TSM is accurate and efficient: it ranks the first place on the Something-Something leaderboard upon publication; on Jetson Nano and Galaxy Note8, it achieves a low latency of 13ms and 35ms for online video recognition.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143019083-abc0de39-9ea1-4175-be5c-073c90de64c3.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| config                                   |   resolution   | gpus | backbone | pretrain | top1 acc | top5 acc | inference_time(video/s) | gpu_mem(M) |                  ckpt                   |                  log                   |
| :--------------------------------------- | :------------: | :--: | :------: | :------: | :------: | :------: | :---------------------: | :--------: | :-------------------------------------: | :------------------------------------: |
| [tsm_r50_1x1x8_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  72.96   |  90.45   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_1x1x8_100e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_100e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.11   |  90.06   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_1x1x16_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.64   |  91.42   |            x            |   27044    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_r50_dense_1x1x8_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.39   |  90.78   |            x            |   13723    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_embedded_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.45   |  91.11   |            x            |   19726    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_dot_product_r50_1x1x8_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  74.17   |  90.95   |            x            |   18413    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |
| [tsm_nl_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_8xb16_kinetics400_rgb.py) | short-side 320 |  8   | ResNet50 | ImageNet |  73.37   |  90.82   |            x            |   19925    | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings. The checkpoints for reference repo can be downloaded [here](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_reference_ckpt.rar).
4. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip), which is referred from the [original repo](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd).
   We use efficient setting as default provided in config files, and it can be changed to accurate setting by

```python
...
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        # `num_clips = 8` when using 8 segments
        num_clips=16,
        # set `twice_sample=True` for twice sample in accurate setting
        twice_sample=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224), it is used for efficient setting
    dict(type='ThreeCrop', crop_size=256),  # it is used for accurate setting
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
```

5. When applying Mixup and CutMix, we use the hyper parameter `alpha=0.2`.
6. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.
7. The **infer_ckpt** means those checkpoints are ported from [TSM](https://github.com/mit-han-lab/temporal-shift-module/blob/master/test_models.py).

:::

For more details on data preparation, you can refer to corresponding parts in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option.

```shell
python tools/train.py configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset.

```shell
python tools/test.py configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
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
