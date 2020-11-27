# TSM

## Introduction
```
@inproceedings{lin2019tsm,
  title={TSM: Temporal Shift Module for Efficient Video Understanding},
  author={Lin, Ji and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2019}
}

@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone | pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |70.24|89.56|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7079 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/20200607_211800.log.json)|
|[tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.59|89.52|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/20200725_031623.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x8_50e_kinetics400_rgb/20200725_031623.log.json)|
|[tsm_r50_video_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_video_1x1x8_50e_kinetics400_rgb.py) |short-side 256|8| ResNet50| ImageNet |70.25|89.66|[70.36](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|[89.49](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_8f.sh)|74.0 (8x1 frames)| 7077 | [ckpt]( https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_1x1x8_100e_kinetics400_rgb_20200702-a77f4328.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_video_1x1x8_100e_kinetics400_rgb/tsm_r50_video_2d_1x1x8_50e_kinetics400_rgb.log.json)|
|[tsm_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py) |340x256|8x4| ResNet50 | ImageNet|72.9|90.44|[72.22](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#dense-sample)|[90.37](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#dense-sample)|11.5 (8x10 frames)| 7079 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/tsm_r50_dense_1x1x8_100e_kinetics400_rgb_20200626-91a54551.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20200626_213415.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb/20200626_213415.log.json)|
|[tsm_r50_dense_1x1x8_100e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_dense_1x1x8_100e_kinetics400_rgb.py) |short-side 256|8| ResNet50 | ImageNet|73.38|91.02|x|x|x|7079|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb_20200727-e1e0c785.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb/20200725_032043.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_dense_256p_1x1x8_100e_kinetics400_rgb/20200725_032043.log.json)|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) |340x256|8| ResNet50| ImageNet |72.09|90.37|[70.67](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|[89.98](https://github.com/mit-han-lab/temporal-shift-module/blob/8d53d6fda40bea2f1b37a6095279c4b454d672bd/scripts/train_tsm_kinetics_rgb_16f.sh)|47.0 (16x1 frames)| 10404  | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/tsm_r50_340x256_1x1x16_50e_kinetics400_rgb_20201011-2f27f229.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20201011_205356.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb/20201011_205356.log.json)|
|[tsm_r50_1x1x16_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_kinetics400_rgb.py) |short-side 256|8x4| ResNet50| ImageNet |71.89|90.73|x|x|x|10398|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/tsm_r50_256p_1x1x16_50e_kinetics400_rgb_20201010-85645c2a.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/20201010_224825.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_256p_1x1x16_50e_kinetics400_rgb/20201010_224825.log.json)|
|[tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4| ResNet50| ImageNet |72.03|90.25|71.81|90.36|x|8931|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb_20200724-f00f1336.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200724_120023.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_embedded_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200724_120023.log.json)|
|[tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4| ResNet50| ImageNet |70.70|89.90|x|x|x|10125|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb_20200816-b93fd297.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200815_210253.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_gaussian_r50_1x1x8_50e_kinetics400_rgb/20200815_210253.log.json)|
|[tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 320|8x4|ResNet50| ImageNet |71.60|90.34|x|x|x|8358|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb_20200724-d8ad84d2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/20200723_220442.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_nl_dot_product_r50_1x1x8_50e_kinetics400_rgb/20200723_220442.log.json)|

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| reference top1 acc (efficient/accurate)| reference top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py) |height 100|8| ResNet50 | ImageNet|45.46 / 47.21|74.71 / 76.09|[45.50 / 47.33](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[74.34 / 76.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7077| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/tsm_r50_1x1x8_50e_sthv1_rgb_20200616-3417f361.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20200616_022852.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb/20200616_022852.log.json)|
|[tsm_r50_1x1x16_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb.py)|height 100|8| ResNet50 | ImageNet|47.62 / 49.28|76.63 / 77.82|[47.05 / 48.61](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[76.40 / 77.96](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|10390|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/tsm_r50_1x1x16_50e_sthv1_rgb_20201010-17fa49f6.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/20201010_221240.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv1_rgb/20201010_221240.log.json)|
|[tsm_r101_1x1x8_50e_sthv1_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb.py)|height 100|8| ResNet50 | ImageNet|45.72 / 48.43|74.67 / 76.72|[46.64 / 48.13](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[75.40 / 77.31](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|9800|[ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/tsm_r101_1x1x8_50e_sthv1_rgb_20201010-43fedf2e.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/20201010_224055.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv1_rgb/20201010_224055.log.json)|

### Something-Something V2

|config | resolution | gpus | backbone | pretrain| top1 acc (efficient/accurate)| top5 acc (efficient/accurate)|  reference top1 acc (efficient/accurate)| reference top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tsm_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb.py) |height 240|8| ResNet50| ImageNet |57.86 / 61.12|84.67 / 86.26|[57.98 / 60.69](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[84.57 / 86.28](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 7069 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/tsm_r50_1x1x8_50e_sthv2_rgb_20200912-033c4ac6.pth)|[log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/20200912_140737.log)|[json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_sthv2_rgb/20200912_140737.log.json)|
|[tsm_r50_1x1x16_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb.py) |height 240|8| ResNet50| ImageNet |59.93 / 62.04|86.10 / 87.35|[58.90 / 60.98](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[85.29 / 86.60](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 10400| [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/tsm_r50_1x1x16_50e_sthv2_rgb_20201010-16469c6f.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20201010_224215.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x16_50e_sthv2_rgb/20201010_224215.log.json)|
|[tsm_r101_1x1x8_50e_sthv2_rgb](/configs/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb.py) |height 240|8| ResNet101 | ImageNet|58.59 / 61.51|85.07 / 86.90|[58.89 / 61.36](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)|[85.14 / 87.00](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd#training)| 9784 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/tsm_r101_1x1x8_50e_sthv2_rgb_20201010-98cdedb8.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20201010_224100.log)| [json](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r101_1x1x8_50e_sthv2_rgb/20201010_224100.log.json)|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings. The checkpoints for reference repo can be downloaded [here](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_reference_ckpt.rar).
4. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop * 1 clip) and accurate setting (Three crop * 2 clip), which is referred from the [original repo](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd).
We use efficient setting as default provided in config files, and it can be changed to accurate setting by
```python
...
test_cfg = dict(average_clips='prob')
...
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,   # `num_clips = 8` when using 8 segments
        twice_sample=True,    # set `twice_sample=True` for twice sample in accurate setting
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224), it is used for efficient setting
    dict(type='ThreeCrop', crop_size=256),  # it is used for accurate setting
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
```

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](/docs/data_preparation.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TSM model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py \
    --work-dir work_dirs/tsm_r50_1x1x8_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TSM model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
