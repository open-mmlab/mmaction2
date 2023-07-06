# ResNet for Audio

[Audiovisual SlowFast Networks for Video Recognition](https://arxiv.org/abs/2001.08740)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We present Audiovisual SlowFast Networks, an architecture for integrated audiovisual perception. AVSlowFast has Slow and Fast visual pathways that are deeply inte- grated with a Faster Audio pathway to model vision and sound in a unified representation. We fuse audio and vi- sual features at multiple layers, enabling audio to con- tribute to the formation of hierarchical audiovisual con- cepts. To overcome training difficulties that arise from dif- ferent learning dynamics for audio and visual modalities, we introduce DropPathway, which randomly drops the Au- dio pathway during training as an effective regularization technique. Inspired by prior studies in neuroscience, we perform hierarchical audiovisual synchronization to learn joint audiovisual features. We report state-of-the-art results on six video action classification and detection datasets, perform detailed ablation studies, and show the gener- alization of AVSlowFast to learn self-supervised audiovi- sual features. Code will be made available at: https: //github.com/facebookresearch/SlowFast.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/30782254/147050415-a30ad32a-ce52-452d-ac3d-91058c8d0cc9.png" width="800"/>
</div>

## Results and Models

### Kinetics-400

| frame sampling strategy | n_fft | gpus | backbone | pretrain | top1 acc | top5 acc | testing protocol | gpu_mem(M) |                 config                 |                 ckpt                 |                 log                  |
| :---------------------: | :---: | :--: | :------: | :------: | :------: | :------: | :--------------: | :--------: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
|         64x1x1          | 1024  |  8   | Resnet18 |   None   |   19.7   |  35.75   |     10 clips     |    1897    | [config](/configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature_20201012-bf34df6c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.log) |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. The validation set of Kinetics400 we used consists of 19796 videos. These videos are available at [Kinetics400-Validation](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB). The corresponding [data list](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (each line is of the format 'video_id, num_frames, label_index') and the [label map](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) are also available.

For more details on data preparation, you can refer to `Prepare audio` in [Data Preparation Tutorial](/docs/en/user_guides/prepare_dataset.md).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train ResNet model on Kinetics-400 audio dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
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
