# TimeSformer

## Introduction

<!-- [ALGORITHM] -->

```BibTeX
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?},
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

## Model Zoo

### Kinetics-400

|config | resolution | gpus | backbone | pretrain | top1 acc| top5 acc | inference_time(video/s) | gpu_mem(M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[timesformer_divST_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py) | short-side 320 | 8 | TimeSformer | ImageNet-21K | 77.92 | 93.29 | x | 17874 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb/timesformer_divST_8x32x1_15e_kinetics400_rgb.json)|
|[timesformer_jointST_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb.py) | short-side 320 | 8 | TimeSformer | ImageNet-21K | 77.01 | 93.08 | x | 25658 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb-0d6e3984.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_jointST_8x32x1_15e_kinetics400_rgb/timesformer_jointST_8x32x1_15e_kinetics400_rgb.json)|
|[timesformer_sapceOnly_8x32x1_15e_kinetics400_rgb](/configs/recognition/timesformer/timesformer_sapceOnly_8x32x1_15e_kinetics400_rgb.py) | short-side 320 | 8 | TimeSformer | ImageNet-21K | 76.93 | 92.90 | x | 12750 | [ckpt](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb-0cf829cd.pth) | [log](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb.log)| [json](https://download.openmmlab.com/mmaction/recognition/timesformer/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb/timesformer_spaceOnly_8x32x1_15e_kinetics400_rgb.json)|

Notes:

1. The **gpus** indicates the number of gpu (32G V100) we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.005 for 8 GPUs x 8 videos/gpu and lr=0.00375 for 8 GPUs x 6 videos/gpu.
2. We keep the test setting with the [original repo](https://github.com/facebookresearch/TimeSformer) (three crop x 1 clip).
3. The pretrained model `vit_base_patch16_224.pth` used by TimeSformer was converted from [vision_transformer](https://github.com/google-research/vision_transformer).

For more details on data preparation, you can refer to Kinetics400 in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TimeSformer model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --work-dir work_dirs/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TimeSformer model on Kinetics-400 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/timesformer/timesformer_divST_8x32x1_15e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
