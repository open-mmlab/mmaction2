# X3D

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@misc{feichtenhofer2020x3d,
      title={X3D: Expanding Architectures for Efficient Video Recognition},
      author={Christoph Feichtenhofer},
      year={2020},
      eprint={2004.04730},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 模型库

### Kinetics-400

|配置文件 | 分辨率 | 主干网络 | top1 10-view | top1 30-view | 参考代码的 top1 10-view | 参考代码的 top1 30-view | ckpt |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|[x3d_s_13x6x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py)|短边 320| X3D_S | 72.7 | 73.2 | 73.1 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | 73.5 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | [ckpt](https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_s_facebook_13x6x1_kinetics400_rgb_20201027-623825a0.pth)[1] |
|[x3d_m_16x5x1_facebook_kinetics400_rgb](/configs/recognition/x3d/x3d_m_16x5x1_facebook_kinetics400_rgb.py)|短边 320| X3D_M | 75.0 | 75.6 | 75.1 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | 76.2 [[SlowFast](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md)] | [ckpt](https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth)[1] |

[1] 这里的模型是从 [SlowFast](https://github.com/facebookresearch/SlowFast/) 代码库中导入并在 MMAction2 使用的数据上进行测试的。目前仅支持 X3D 模型的测试，训练部分将会在近期提供。

注：

1. 参考代码的结果是通过使用相同的数据和原来的代码库所提供的模型进行测试得到的。
2. 我们使用的 Kinetics400 验证集包含 19796 个视频，用户可以从 [验证集视频](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EbXw2WX94J1Hunyt3MWNDJUBz-nHvQYhO9pvKqm6g39PMA?e=a9QldB) 下载这些视频。同时也提供了对应的 [数据列表](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_val_list.txt) （每行格式为：视频 ID，视频帧数目，类别序号）以及 [标签映射](https://download.openmmlab.com/mmaction/dataset/k400_val/kinetics_class2ind.txt) （类别序号到类别名称）。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 Kinetics400 部分

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 Kinetics-400 数据集上测试 X3D 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/recognition/x3d/x3d_s_13x6x1_facebook_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json --average-clips prob
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
