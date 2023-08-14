# UniFormer

[UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning](https://arxiv.org/abs/2201.04676)

<!-- [ALGORITHM] -->

## 简介

```BibTeX
@inproceedings{
  li2022uniformer,
  title={UniFormer: Unified Transformer for Efficient Spatial-Temporal Representation Learning},
  author={Kunchang Li and Yali Wang and Gao Peng and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=nBU_u6DLvoK}
}
```

## 模型库

### Kinetics-400

| 帧采样策略 |     分辨率     |  主干网络   | top1 准确率 | top5 准确率 | [参考文献](https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md) top1 准确率 | [参考文献](https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md) top5 准确率 | mm-Kinetics top1 准确率 | mm-Kinetics top5 准确率 |     测试方案     | FLOPs | 参数量 |                                             配置文件                                              |                                                                           ckpt                                                                           |
| :--------: | :------------: | :---------: | :---------: | :---------: | :---------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: | :---------------------: | :---------------------: | :--------------: | :---: | :----: | :-----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   16x4x1   | short-side 320 | UniFormer-S |    80.9     |    94.6     |                                                 80.8                                                  |                                                 94.7                                                  |          80.9           |          94.6           | 4 clips x 1 crop | 41.8G | 21.4M  | [config](/configs/recognition/uniformer/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-c630a037.pth) |
|   16x4x1   | short-side 320 | UniFormer-B |    82.0     |    95.0     |                                                 82.0                                                  |                                                 95.1                                                  |          82.0           |          95.0           | 4 clips x 1 crop | 96.7G | 49.8M  | [config](/configs/recognition/uniformer/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-base_imagenet1k-pre_16x4x1_kinetics400-rgb_20221219-157c2e66.pth)  |
|   32x4x1   | short-side 320 | UniFormer-B |    83.1     |    95.3     |                                                 82.9                                                  |                                                 95.4                                                  |          83.0           |          95.3           | 4 clips x 1 crop |  59G  | 49.8M  | [config](/configs/recognition/uniformer/uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb.py)  | [ckpt](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv1/uniformer-base_imagenet1k-pre_32x4x1_kinetics400-rgb_20221219-b776322c.pth)  |

这些模型迁移自 [UniFormer](https://github.com/Sense-X/UniFormer/blob/main/video_classification/README.md)仓库，并在我们的数据上进行了测试。目前，我们仅支持对 UniFormer 模型的测试，训练功能将很快提供。

1. 名称为"参考文献"的列中的值是原始仓库的结果。
2. `top1/5 准确率`中的值是模型在与原始仓库相同的数据集上的测试结果，分类器结果-标签映射与[UniFormer](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL)一致。数据集总共有19787个视频，可以在[Kinetics400](https://pan.baidu.com/s/1t5K0FRz3PGAT-37-3FwAfg)（百度云密码：g5kp）中获取。
3. 名称为 "mm-Kinetics" 的列中的值是模型在 MMAction2 持有的 Kinetics 数据集上的测试结果，其他 MMAction2 模型也使用了该数据集。由于 Kinetics 数据集的各个版本之间存在差异，因此 `top1/5 准确率` 和 `mm-Kinetics top1/5 准确率` 之间存在一些差距。为了与其他模型进行公平比较，我们在这里报告了两个结果。请注意，我们只报告了推理结果，由于 UniFormer 和其他模型之间的训练集不同，该结果低于在作者版本上测试的结果。
4. 由于 Kinetics-400/600/700 的原始模型采用了不同的[标签文件](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL)，我们根据标签名称简单地映射了权重。Kinetics-400/600/700 的新标签映射可以在[这里](https://github.com/open-mmlab/mmaction2/tree/main/tools/data/kinetics)找到。
5. 由于 \[SlowFast\] (https://github.com/facebookresearch/SlowFast)和 MMAction2 之间存在一些差异，它们的性能存在一些差距。

有关数据准备的更多详细信息，您可以参考[准备_kinetics](/tools/data/kinetics/README_zh-CN.md)。

## 如何测试

您可以使用以下命令来测试模型：

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

示例：在Kinetics-400数据集上测试 UniFormer-S 模型，并将结果转储到一个 pkl 文件中。

```shell
python tools/test.py configs/recognition/uniformer/uniformer-small_imagenet1k-pre_16x4x1_kinetics400-rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

有关更多详细信息，请参考[训练和测试教程](/docs/zh_cn/user_guides/train_test.md)中的**测试**部分。
