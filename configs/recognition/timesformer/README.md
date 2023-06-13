# TimeSformer

[Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named "TimeSformer," adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of frame-level patches. Our experimental study compares different self-attention schemes and suggests that "divided attention," where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered. Despite the radically new design, TimeSformer achieves state-of-the-art results on several action recognition benchmarks, including the best reported accuracy on Kinetics-400 and Kinetics-600. Finally, compared to 3D convolutional networks, our model is faster to train, it can achieve dramatically higher test efficiency (at a small drop in accuracy), and it can also be applied to much longer video clips (over one minute long).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143018542-7f782ec9-dca2-495e-9043-c13ad941a25c.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | resolution | gpus |        backbone         |   pretrain   | top1 acc | top5 acc | testing protocol | FLOPs | params |             config             |             ckpt             |             log             |
| :---------------------: | :--------: | :--: | :---------------------: | :----------: | :------: | :------: | :--------------: | :---: | :----: | :----------------------------: | :--------------------------: | :-------------------------: |
|         8x32x1          |  224x224   |  8   |   TimeSformer (divST)   | ImageNet-21K |  77.69   |  93.45   | 1 clip x 3 crop  | 196G  |  122M  | [config](/configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.log) |
|         8x32x1          |  224x224   |  8   |  TimeSformer (jointST)  | ImageNet-21K |  76.95   |  93.28   | 1 clip x 3 crop  | 180G  | 86.11M | [config](/configs/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-8022d1c0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_jointST_8xb8-8x32x1-15e_kinetics400-rgb.log) |
|         8x32x1          |  224x224   |  8   | TimeSformer (spaceOnly) | ImageNet-21K |  76.93   |  92.88   | 1 clip x 3 crop  | 141G  | 86.11M | [config](/configs/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb_20220815-78f05367.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition/timesformer/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb/timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size and the original batch size.
2. We keep the test setting with the [original repo](https://github.com/facebookresearch/TimeSformer) (three crop x 1 clip).
3. The pretrained model `vit_base_patch16_224.pth` used by TimeSformer was converted from [vision_transformer](https://github.com/google-research/vision_transformer).

For more details on data preparation, you can refer to [Kinetics400](/tools/data/kinetics/README.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train TimeSformer model on Kinetics-400 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py \
    --seed=0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test TimeSformer model on Kinetics-400 dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition/timesformer/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

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
