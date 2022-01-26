# TRN

[Temporal Relational Reasoning in Videos](https://openaccess.thecvf.com/content_ECCV_2018/html/Bolei_Zhou_Temporal_Relational_Reasoning_ECCV_2018_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Temporal relational reasoning, the ability to link meaningful transformations of objects or entities over time, is a fundamental property of intelligent species. In this paper, we introduce an effective and interpretable network module, the Temporal Relation Network (TRN), designed to learn and reason about temporal dependencies between video frames at multiple time scales. We evaluate TRN-equipped networks on activity recognition tasks using three recent video datasets - Something-Something, Jester, and Charades - which fundamentally depend on temporal relational reasoning. Our results demonstrate that the proposed TRN gives convolutional neural networks a remarkable capacity to discover temporal relations in videos. Through only sparsely sampled video frames, TRN-equipped networks can accurately predict human-object interactions in the Something-Something dataset and identify various human gestures on the Jester dataset with very competitive performance. TRN-equipped networks also outperform two-stream networks and 3D convolution networks in recognizing daily activities in the Charades dataset. Further analyses show that the models learn intuitive and interpretable visual common sense knowledge in videos.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018998-d2120c3d-a9a7-4e4c-90b1-1e5ff1fd5f06.png" width="800"/>
</div>

## Results and Models

### Something-Something V1

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv1_rgb](/configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py) | height 100 | 8 | ResNet50 | ImageNet | 31.62 / 33.88 |60.01 / 62.12| 11010 | [ckpt](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb/trn_r50_1x1x8_50e_sthv1_rgb_20210401-163704a8.pth) | [log](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb/20210326_103948.log)| [json](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb/20210326_103948.log.json)|

### Something-Something V2

|config | resolution | gpus | backbone| pretrain | top1 acc (efficient/accurate)| top5 acc (efficient/accurate)| gpu_mem(M)  | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[trn_r50_1x1x8_50e_sthv2_rgb](/configs/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb.py) | height 256 | 8 | ResNet50 | ImageNet | 48.39 / 51.28 |76.58 / 78.65 | 11010 | [ckpt](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb/trn_r50_1x1x8_50e_sthv2_rgb_20210816-7abbc4c1.pth) | [log](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb/20210816_221356.log)| [json](https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv2_rgb/20210816_221356.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. There are two kinds of test settings for Something-Something dataset, efficient setting (center crop x 1 clip) and accurate setting (Three crop x 2 clip).
3. In the original [repository](https://github.com/zhoubolei/TRN-pytorch), the author augments data with random flipping on something-something dataset, but the augmentation method may be wrong due to the direct actions, such as `push left to right`. So, we replaced `flip` with `flip with label mapping`, and change the testing method `TenCrop`, which has five flipped crops, to `Twice Sample & ThreeCrop`.
4. We use `ResNet50` instead of `BNInception` as the backbone of TRN. When Training `TRN-ResNet50` on sthv1 dataset in the original repository, we get top1 (top5) accuracy 30.542 (58.627) vs. ours 31.62 (60.01).

:::

For more details on data preparation, you can refer to

- [preparing_sthv1](/tools/data/sthv1/README.md)
- [preparing_sthv2](/tools/data/sthv2/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TRN model on sthv1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    --work-dir work_dirs/trn_r50_1x1x8_50e_sthv1_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TRN model on sthv1 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```
