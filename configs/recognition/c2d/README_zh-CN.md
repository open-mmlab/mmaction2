# TSN

## 简介

<!-- [ALGORITHM] -->
C2D 是 [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) 的基准模型

注意: C2D的实现在 1.上述文章; 2."SlowFast"仓库; 3."Video-Nonlocal-Net"仓库； 三者稍有不同

MMAction2 中的 C2D 与 ["Video-Nonlocal-Net"仓库](https://github.com/facebookresearch/video-nonlocal-net/tree/main/scripts/run_c2d_baseline_400k.sh)保持一致

具体地:
- maxpool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
- maxpool3d_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))


MMAction2 中的 C2D_Nopool 与 ["SlowFast"仓库](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml)保持一致



## 模型库

### Kinetics-400

| 配置文件                                                                                                           |     分辨率     | GPU 数量 | 主干网络 |  预训练  | top1 准确率 | top5 准确率 |                                      参考代码的 top1 准确率                                      |                                      参考代码的 top5 准确率                                      | 推理时间 (video/s) | GPU 显存占用 (M) |   ckpt   |   log   |   json   |
| :----------------------------------------------------------------------------------------------------------------- | :------------: | :------: | :------: | :------: | :---------: | :---------: | :----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: | :----------------: | :--------------: | :------: | :-----: | :------: |
| [c2d_nopool_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/c2d/c2d_nopool_r50_8x8x1_100e_kinetics400_rgb.py) | short-side 256 |    8     | ResNet50 | ImageNet |    70.53    |    89.26    | [67.2](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) | [87.8](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md#kinetics-400-and-600) |         x          |      21548       | [ckpt]() | [log]() | [json]() |
| [c2d_r50_8x8x1_100e_kinetics400_rgb](/configs/recognition/c2d/c2d_r50_8x8x1_100e_kinetics400_rgb.py)               | short-side 256 |    8     | ResNet50 | ImageNet |    71.95    |    89.82    | [71.9](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) | [90.0](https://github.com/facebookresearch/video-nonlocal-net#modifications-for-improving-speed) |         x          |      16961       | [ckpt]() | [log]() | [json]() |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 参考代码的结果是通过使用相同的模型配置在原来的代码库上训练得到的。
4. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考：

- [准备 ucf101](/tools/data/ucf101/README_zh-CN.md)
- [准备 kinetics](/tools/data/kinetics/README_zh-CN.md)
- [准备 sthv1](/tools/data/sthv1/README_zh-CN.md)
- [准备 sthv2](/tools/data/sthv2/README_zh-CN.md)
- [准备 mit](/tools/data/mit/README_zh-CN.md)
- [准备 mmit](/tools/data/mmit/README_zh-CN.md)
- [准备 hvu](/tools/data/hvu/README_zh-CN.md)
- [准备 hmdb51](/tools/data/hmdb51/README_zh-CN.md)

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

<!-- 例如：以一个确定性的训练方式，辅以定期的验证过程进行 TSN 模型在 Kinetics-400 数据集上的训练。

```shell
python tools/train.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    --work-dir work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
``` -->

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#%E8%AE%AD%E7%BB%83%E9%85%8D%E7%BD%AE) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

<!-- 例如：在 Kinetics-400 数据集上测试 TSN 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
``` -->

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#%E6%B5%8B%E8%AF%95%E6%9F%90%E4%B8%AA%E6%95%B0%E6%8D%AE%E9%9B%86) 中的 **测试某个数据集** 部分。



## 引用

```BibTeX
@article{XiaolongWang2017NonlocalNN,
  title={Non-local Neural Networks},
  author={Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2017}
}
```