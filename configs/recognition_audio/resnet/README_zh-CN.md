# ResNet for Audio

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@article{xiao2020audiovisual,
  title={Audiovisual SlowFast Networks for Video Recognition},
  author={Xiao, Fanyi and Lee, Yong Jae and Grauman, Kristen and Malik, Jitendra and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2001.08740},
  year={2020}
}
```

## 模型库

### Kinetics-400

| 配置文件                                                                                                                                                                                                                                                         | n_fft | GPU 数量 |   主干网络    | 预训练 | top1 acc/delta | top5 acc/delta | 推理时间 (video/s) | GPU 显存占用 (M) |                                                                                              ckpt                                                                                               |                                                                      log                                                                       |                                                                         json                                                                         |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---: | :------: | :-----------: | :----: | :------------: | :------------: | :----------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| [tsn_r18_64x1x1_100e_kinetics400_audio_feature](/configs/recognition_audio/resnet/tsn_r18_64x1x1_100e_kinetics400_audio_feature.py)                                                                                                                              | 1024  |    8     |   ResNet18    |  None  |      19.7      |     35.75      |         x          |       1897       | [ckpt](https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/tsn_r18_64x1x1_100e_kinetics400_audio_feature_20201012-bf34df6c.pth) | [log](https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/20201010_144630.log) | [json](https://download.openmmlab.com/mmaction/recognition/audio_recognition/tsn_r18_64x1x1_100e_kinetics400_audio_feature/20201010_144630.log.json) |
| [tsn_r18_64x1x1_100e_kinetics400_audio_feature](/configs/recognition_audio/resnet/tsn_r18_64x1x1_100e_kinetics400_audio_feature.py) + [tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb.py) | 1024  |    8     | ResNet(18+50) |  None  |  71.50(+0.39)  |  90.18(+0.14)  |         x          |        x         |                                                                                                x                                                                                                |                                                                       x                                                                        |                                                                          x                                                                           |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs/zh_cn/data_preparation.md) 中的准备音频部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: 以一个确定性的训练方式，辅以定期的验证过程进行 ResNet 模型在 Kinetics400 音频数据集上的训练。

```shell
python tools/train.py configs/audio_recognition/tsn_r50_64x1x1_100e_kinetics400_audio_feature.py \
    --work-dir work_dirs/tsn_r50_64x1x1_100e_kinetics400_audio_feature \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 音频数据集上测试 ResNet 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/audio_recognition/tsn_r50_64x1x1_100e_kinetics400_audio_feature.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。

## 融合

对于多模态融合，用户可以使用这个 [脚本](/tools/analysis/report_accuracy.py)，其命令大致为：

```shell
python tools/analysis/report_accuracy.py --scores ${AUDIO_RESULT_PKL} ${VISUAL_RESULT_PKL} --datalist data/kinetics400/kinetics400_val_list_rawframes.txt --coefficient 1 1
```

- AUDIO_RESULT_PKL: `tools/test.py` 脚本通过 `--out` 选项存储的输出文件。
- VISUAL_RESULT_PKL: `tools/test.py` 脚本通过 `--out` 选项存储的输出文件。
