# I3D

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

<!-- [BACKBONE] -->

```BibTeX
@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | GPU 数量 | 主干网络 |预训练| top1 准确率| top5 准确率 | 推理时间 (video/s) | GPU 显存占用 (M)| ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |340x256|8| ResNet50|ImageNet |72.68|90.78|1.7 (320x3 frames)| 5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb/20200614_060456.log.json)|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |短边 256|8| ResNet50|ImageNet | 73.27|90.92|x|5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_256p_32x2x1_100e_kinetics400_rgb_20200801-7d9f44de.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/20200725_031555.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_256p_32x2x1_100e_kinetics400_rgb/20200725_031555.log.json)|
|[i3d_r50_video_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb.py)|短边 256p|8| ResNet50 |ImageNet|72.85 |90.75 |x|5170|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/20200706_143014.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/20200706_143014.log.json)|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) |340x256|8x2| ResNet50| ImageNet|72.77|90.57|1.7 (320x3 frames)| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_32x2x1_100e_kinetics400_rgb_20200616-2bbb4361.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb/20200616_230011.log.json)|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) |短边 256|8| ResNet50| ImageNet|73.48|91.00|x| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb_20200725-24eb54cc.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/20200725_031604.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_dense_256p_32x2x1_100e_kinetics400_rgb/20200725_031604.log.json)|
|[i3d_r50_lazy_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_lazy_32x2x1_100e_kinetics400_rgb.py) |340x256|8| ResNet50 |ImageNet|72.32|90.72|1.8 (320x3 frames)| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/i3d_r50_fast_32x2x1_100e_kinetics400_rgb_20200612-000e4d2a.pth) | [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log)| [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_32x2x1_100e_kinetics400_rgb/20200612_233836.log.json)|
|[i3d_r50_lazy_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_lazy_32x2x1_100e_kinetics400_rgb.py) |短边 256|8| ResNet50| ImageNet|73.24|90.99|x| 5170| [ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb_20200817-4e90d1d5.pth)| [log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/20200725_031457.log) | [json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_fast_256p_32x2x1_100e_kinetics400_rgb/20200725_031457.log.json) |
|[i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb.py)|短边 256p|8x4| ResNet50 |ImageNet|74.71|91.81|x|6438|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200813-6e6aef1b.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034054.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_embedded_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034054.log.json)|
|[i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb.py)|短边 256p|8x4| ResNet50 |ImageNet|73.37|91.26|x|4944|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb_20200815-17f84aa2.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034909.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_gaussian_r50_32x2x1_100e_kinetics400_rgb/20200813_034909.log.json)|
|[i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb.py)|短边 256p|8x4| ResNet50 |ImageNet|73.92|91.59|x|4832|[ckpt](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb_20200814-7c30d5bb.pth)|[log](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/20200814_044208.log)|[json](https://download.openmmlab.com/mmaction/recognition/i3d/i3d_nl_dot_product_r50_32x2x1_100e_kinetics400_rgb/20200814_044208.log.json)|

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 这里的 **推理时间** 是根据 [基准测试脚本](/tools/analysis/benchmark.py) 获得的，采用测试时的采帧策略，且只考虑模型的推理时间，
   并不包括 IO 时间以及预处理时间。对于每个配置，MMAction2 使用 1 块 GPU 并设置批大小（每块 GPU 处理的视频个数）为 1 来计算推理时间。
3. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400 部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 I3D 模型在 Kinetics400 数据集上的训练。

```shell
python tools/train.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    --work-dir work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics400 数据集上测试 I3D 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
