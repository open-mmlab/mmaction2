# TIN

## 简介

[ALGORITHM]

```BibTeX
@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```

## 模型库

### Something-Something V1

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv1_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py)|高 100|8x4| ResNet50 | ImageNet | 44.25 | 73.94 | 44.04 | 72.72 | 6181 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/tin_r50_1x1x8_40e_sthv1_rgb_20200729-4a33db86.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb/20200729_034132.log.json) |

### Something-Something V2

|配置文件 | 分辨率 | GPU 数量 | 主干网络 | 预训练 | top1 准确率 | top5 准确率 | 参考代码的 top1 准确率 | 参考代码的 top5 准确率 | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_r50_1x1x8_40e_sthv2_rgb](/configs/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb.py)|高 240|8x4| ResNet50 | ImageNet | 56.70 | 83.62 | 56.48 | 83.45 | 6185 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/tin_r50_1x1x8_40e_sthv2_rgb_20200912-b27a7337.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_r50_1x1x8_40e_sthv2_rgb/20200912_225451.log.json) |

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络| 预训练 | top1 准确率| top5 准确率 | GPU 显存占用 (M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb.py)|短边 256|8x4| ResNet50 | TSM-Kinetics400 | 70.89 | 89.89 | 6187 | [ckpt](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb_20200810-4a146a70.pth) | [log](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log) | [json](https://download.openmmlab.com/mmaction/recognition/tin/tin_tsm_finetune_r50_1x1x8_50e_kinetics400_rgb/20200809_142447.log.json) |

Here, we use `finetune` to indicate that we use [TSM model](https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth) trained on Kinetics-400 to finetune the TIN model on Kinetics-400.

Notes:

1. The **reference topk acc** are got by training the [original repo #1aacd0c](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) with no [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4).
   The [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4) will lead to incorrect performance, so we fix it before running.
2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
3. The **inference_time** is got by this [benchmark script](/tools/analysis/benchmark.py), where we use the sampling frames strategy of the test setting and only care about the model inference time,
   not including the IO time and pre-processing time. For each setting, we use 1 gpu and set batch size (videos per gpu) to 1 to calculate the inference time.
4. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.

For more details on data preparation, you can refer to Kinetics400, Something-Something V1 and Something-Something V2 in [Data Preparation](/docs/data_preparation.md).

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 TIN 模型在 Something-Something V1 数据集上的训练。

```shell
python tools/train.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    --work-dir work_dirs/tin_r50_1x1x8_40e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Something-Something V1 数据集上测试 TIN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tin/tin_r50_1x1x8_40e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
