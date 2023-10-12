# ResNet for Audio

[Audiovisual SlowFast Networks for Video Recognition](https://arxiv.org/abs/2001.08740)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present Audiovisual SlowFast Networks, an architecture for integrated audiovisual perception. AVSlowFast has Slow and Fast visual pathways that are deeply integrated with a Faster Audio pathway to model vision and sound in a unified representation. We fuse audio and visual features at multiple layers, enabling audio to contribute to the formation of hierarchical audiovisual concepts. To overcome training difficulties that arise from different learning dynamics for audio and visual modalities, we introduce DropPathway, which randomly drops the Au- dio pathway during training as an effective regularization technique. Inspired by prior studies in neuroscience, we perform hierarchical audiovisual synchronization to learn joint audiovisual features. We report state-of-the-art results on six video action classification and detection datasets, perform detailed ablation studies, and show the generalization of AVSlowFast to learn self-supervised audiovisual features.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/30782254/147050415-a30ad32a-ce52-452d-ac3d-91058c8d0cc9.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | n_fft | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol | FLOPs | params |                 config                 |                 ckpt                 |                 log                 |
| :---------------------: | :---: | :--: | :------: | :------: | :------: | :------: | :--------------: | :---: | :----: | :------------------------------------: | :----------------------------------: | :---------------------------------: |
|         64x1x1          | 1024  |  8   | Resnet18 |   None   |   13.7   |   27.3   |     1 clips      | 0.37G | 11.4M  | [config](/configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature_20230702-e4642fb0.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.log) |

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ResNet model on Kinetics-400 audio dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test ResNet model on Kinetics-400 audio dataset and dump the result to a pkl file.

```shell
python tools/test.py configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@article{xiao2020audiovisual,
  title={Audiovisual SlowFast Networks for Video Recognition},
  author={Xiao, Fanyi and Lee, Yong Jae and Grauman, Kristen and Malik, Jitendra and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2001.08740},
  year={2020}
}
```
