# TIN

## Introduction
```
@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```

## Model Zoo

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv1_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py)|height 100|8x4| ResNet50 | ImageNet | 44.25 | 73.94 | 44.04 | 72.72 | 6181 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/tin_r50_1x1x8_40e_sthv1_rgb_20200729-4a33db86.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log.json) |

### Something-Something V2

|config | resolution | gpus | backbone| pretrain | top1 acc| top5 acc | reference top1 acc | reference top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv2_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py)|height 240|8x4| ResNet50 | ImageNet | 56.70 | 83.62 | 56.48 | 83.45 | 6185 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/tin_r50_1x1x8_40e_sthv2_rgb_20200912-b27a7337.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log.json) |

### Kinetics-400

|config | resolution | gpus | backbone| pretrain | top1 acc| top5 acc | gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb.py)|short-side 256|8x4| ResNet50 | TSM-Kinetics400 | 70.89 | 89.89 | 6187 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb_20200810-4a146a70.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log.json) |

Here, we use `finetune` to indicate that we use [TSM model](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) trained on Kinetics-400 to finetune the TIN model on Kinetics-400.

Notes:
1. The **reference topk acc** are got by training the [original repo #1aacd0c](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) with no [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4).
The [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4) will lead to incorrect performance, so we fix it before running.
2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
3. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
4. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](/docs/data_preparation.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TIN model on Something-Something V1 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TIN model on Something-Something V1 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
