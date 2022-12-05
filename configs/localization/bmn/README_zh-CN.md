# BMN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{lin2019bmn,
  title={Bmn: Boundary-matching network for temporal action proposal generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3889--3898},
  year={2019}
}
```

<!-- [DATASET] -->

```BibTeX
@article{zhao2017cuhk,
  title={Cuhk \& ethz \& siat submission to activitynet challenge 2017},
  author={Zhao, Y and Zhang, B and Wu, Z and Yang, S and Zhou, L and Yan, S and Wang, L and Xiong, Y and Lin, D and Qiao, Y and others},
  journal={arXiv preprint arXiv:1710.08011},
  volume={8},
  year={2017}
}
```

## 模型库

### ActivityNet feature

|                                                   配置文件                                                    |      特征      | GPU 数量 | AR@100 |  AUC  | AP@0.5 | AP@0.75 | AP@0.95 |  mAP  | GPU 显存占用 (M) | 推理时间 (s) |                                                                             ckpt                                                                             |                                                                       log                                                                        | json                                                                                                                                               |
| :-----------------------------------------------------------------------------------------------------------: | :------------: | :------: | :----: | :---: | :----: | :-----: | :-----: | :---: | :--------------: | ------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| [bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) | cuhk_mean_100  |    2     | 75.28  | 67.22 | 42.47  |  31.31  |  9.92   | 30.34 |       5420       | 3.27         | [ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature_20200619-42a3b111.pth) |    [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log)     | [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log.json)    |
|                                                                                                               | mmaction_video |    2     | 75.43  | 67.22 | 42.62  |  31.56  |  10.86  | 30.77 |       5420       | 3.27         |  [ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809-c9fd14d2.pth)  | [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.log) | [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.json) |
|                                                                                                               | mmaction_clip  |    2     | 75.35  | 67.38 | 43.08  |  32.19  |  10.73  | 31.15 |       5420       | 3.27         |   [ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809-10d803ce.pth)   |  [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.log)  | [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.json)   |
|           [BMN-official](https://github.com/JJBOY/BMN-Boundary-Matching-Network) (for reference)\*            | cuhk_mean_100  |    -     | 75.27  | 67.49 | 42.22  |  30.98  |  9.22   | 30.00 |        -         | -            |                                                                              -                                                                               |                                                                        -                                                                         | -                                                                                                                                                  |

- 注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 对于 **特征** 这一列，`cuhk_mean_100` 表示所使用的特征为利用 [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) 代码库抽取的，被广泛利用的 CUHK ActivityNet 特征，
   `mmaction_video` 和 `mmaction_clip` 分布表示所使用的特征为利用 MMAction 抽取的，视频级别 ActivityNet 预训练模型的特征；视频片段级别 ActivityNet 预训练模型的特征。
3. MMAction2 使用 ActivityNet2017 未剪辑视频分类赛道上 [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) 所提交的结果来为每个视频的时序动作候选指定标签，以用于 BMN 模型评估。

\*MMAction2 在 [原始代码库](https://github.com/JJBOY/BMN-Boundary-Matching-Network) 上训练 BMN，并且在 [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) 的对应标签上评估时序动作候选生成和时序检测的结果。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs/zh_cn/data_preparation.md) 中的 ActivityNet 特征部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：在 ActivityNet 特征上训练 BMN。

```shell
python tools/train.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 ActivityNet 特征上测试 BMN 模型。

```shell
# 注：如果需要进行指标验证，需确测试数据的保标注文件包含真实标签
python tools/test.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
```

用户也可以利用 [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) 的预测文件评估模型时序检测的结果，并生成时序动作候选文件（即命令中的 `results.json`）

```shell
python tools/analysis/report_map.py --proposal path/to/proposal_file
```

注：

1. (可选项) 用户可以使用以下指令生成格式化的时序动作候选文件，该文件可被送入动作识别器中（目前只支持 SSN 和 P-GCN，不包括 TSN, I3D 等），以获得时序动作候选的分类结果。

   ```shell
   python tools/data/activitynet/convert_proposal_format.py
   ```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
