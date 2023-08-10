# VideoMAE

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## 模型库

### Kinetics-400

| 帧采样策略 |  分辨率  | 主干网络 | top1 准确率 | top5 准确率 |       参考代码的 top1 准确率        |       参考代码的 top5准确率        |     测试协议      | 浮点运算数 | 参数量 |         配置文件          |           ckpt            |
| :--------: | :------: | :------: | :---------: | :---------: | :---------------------------------: | :--------------------------------: | :---------------: | :--------: | :----: | :-----------------------: | :-----------------------: |
|   16x4x1   | 短边 320 |  ViT-B   |    81.3     |    95.0     | 81.5 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 95.1 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops |    180G    |  87M   | [config](/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth) \[1\] |
|   16x4x1   | 短边 320 |  ViT-L   |    85.3     |    96.7     | 85.2 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 96.8 \[[VideoMAE](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)\] | 5 clips x 3 crops |    597G    |  305M  | [config](/configs/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth) \[1\] |

\[1\] 该模型移植自 [VideoMAE](https://github.com/MCG-NJU/VideoMAE) 并在我们的数据集上进行测试。目前仅支持VideoMAE模型的测试，训练即将推出。

1. 参考代码的准确率数据来源于原始仓库。
2. 我们使用的 Kinetics400 验证集包含 19796 个视频。 用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB)下载这些视频。 同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) (每行格式为：视频 ID，视频帧数目，类别序号) 以及 [映射标签](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) 。

关于数据处理的更多细节，用户可以参照 [preparing_kinetics](/tools/data/kinetics/README_zh-CN.md).

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 ViT-base 模型，并将结果导出为一个 pkl 文件。

```shell
python tools/test.py configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

更多测试细节，可参考 [训练和测试教程](/docs/zh_cn/user_guides/train_test.md) 中的 **测试** 部分。
