# TIN

[Temporal Interlacing Network](https://ojs.aaai.org/index.php/AAAI/article/view/6872)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

For a long time, the vision community tries to learn the spatio-temporal representation by combining convolutional neural network together with various temporal models, such as the families of Markov chain, optical flow, RNN and temporal convolution. However, these pipelines consume enormous computing resources due to the alternately learning process for spatial and temporal information. One natural question is whether we can embed the temporal information into the spatial one so the information in the two domains can be jointly learned once-only. In this work, we answer this question by presenting a simple yet powerful operator -- temporal interlacing network (TIN). Instead of learning the temporal features, TIN fuses the two kinds of information by interlacing spatial representations from the past to the future, and vice versa. A differentiable interlacing target can be learned to control the interlacing process. In this way, a heavy temporal model is replaced by a simple interlacing operator. We theoretically prove that with a learnable interlacing target, TIN performs equivalently to the regularized temporal convolution network (r-TCN), but gains 4% more accuracy with 6x less latency on 6 challenging benchmarks. These results push the state-of-the-art performances of video understanding by a considerable margin. Not surprising, the ensemble model of the proposed TIN won the 1st place in the ICCV19 - Multi Moments in Time challenge.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018602-d32bd546-e4f5-442c-9173-e4303676efb3.png" width="800"/>
</div>

## Results and Models

### Something-Something V1

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 100 | 8x4  | ResNet50 | ImageNet |  39.68   |  68.55   |       44.04        |       72.72        | 8 clips x 1 crop |            x            |    6181    | [config](/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb_20220913-9b7804d6.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.log) |

### Something-Something V2

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc | reference top1 acc | reference top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |      config       |      ckpt       |      log       |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :----------------: | :--------------: | :---------------------: | :--------: | :---------------: | :-------------: | :------------: |
|          1x1x8          | height 240 | 8x4  | ResNet50 | ImageNet |  54.78   |  82.18   |       56.48        |       83.45        | 8 clips x 1 crop |            x            |    6185    | [config](/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb_20220913-84f9b4b0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.log) |

### Kinetics-400

| frame sampling strategy |   resolution   | gpus | backbone |    pretrain     | top1 acc | top5 acc | testing protocol | inference time(video/s) | gpu_mem(M) |          config           |          ckpt           |           log           |
| :---------------------: | :------------: | :--: | :------: | :-------------: | :------: | :------: | :--------------: | :---------------------: | :--------: | :-----------------------: | :---------------------: | :---------------------: |
|          1x1x8          | short-side 256 | 8x4  | ResNet50 | TSM-Kinetics400 |  71.86   |  90.44   | 8 clips x 1 crop |            x            |    6185    | [config](/configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv2-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb_20220913-7f10d0c0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/tin/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb/tin_kinetics400-pretrained-tsm-r50_1x1x8-50e_kinetics400-rgb.log) |

Here, we use `finetune` to indicate that we use [TSM model](https://download.openmmlab.com/mmaction/v1.0/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb_20220831-64d69186.pth) trained on Kinetics-400 to finetune the TIN model on Kinetics-400.

:::{note}

1. The **reference topk acc** are got by training the [original repo #1aacd0c](https://github.com/deepcs233/TIN/tree/1aacd0c4c30d5e1d334bf023e55b855b59f158db) with no [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4).
   The [AverageMeter issue](https://github.com/deepcs233/TIN/issues/4) will lead to incorrect performance, so we fix it before running.
2. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
3. The values in columns named after "reference" are the results got by training on the original repo, using the same model settings.
4. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/v1.0/dataset/k400_val/kinetics_class2ind.txt) are also available.

:::

For more details on data preparation, you can refer to

- [Kinetics](/tools/data/kinetics/README.md)
- [Something-something V1](/tools/data/sthv1/README.md)
- [Something-something V2](/tools/data/sthv2/README.md)

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TIN model on Something-Something V1 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py \
    --work-dir work_dirs/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TIN model on Something-Something V1 dataset and dump the result to a json file.

```shell
python tools/test.py configs/recognition/tin/tin_imagenet-pretrained-r50_8xb6-1x1x8-40e_sthv1-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.json
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```
