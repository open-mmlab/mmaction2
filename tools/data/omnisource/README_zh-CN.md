# 准备 OmniSource

## 简介

<!-- [DATASET] -->

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```

MMAction2 中发布了 OmniSource 网络数据集的一个子集 (来自论文 [Omni-sourced Webly-supervised Learning for Video Recognition](https://arxiv.org/abs/2003.13042))。
OmniSource 数据集中所有类别均来自 Kinetics-400。MMAction2 所提供的子集包含属于 Mini-Kinetics 数据集 200 类动作的网络数据 (Mini-inetics 数据集由论文 [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/pdf/1712.04851.pdf) 提出)。

MMAction2 提供所有数据源中属于 Mini-Kinetics 200 类动作的数据，这些数据源包含：Kinetics 数据集，Kinetics 原始数据集（未经裁剪的长视频），来自 Google 和 Instagram 的网络图片，来自 Instagram 的网络视频。为获取这一数据集，用户需先填写 [数据申请表](https://docs.google.com/forms/d/e/1FAIpQLSd8_GlmHzG8FcDbW-OEu__G7qLgOSYZpH-i5vYVJcu7wcb_TQ/viewform?usp=sf_link)。在接收到申请后，下载链接将被发送至用户邮箱。由于发布的数据集均为爬取所得的原始数据，数据集较大，下载需要一定时间。下表中提供了 OmniSource 数据集各个分量的统计信息。

|   数据集名称    | 样本个数 | 所占空间 | 过滤使用的 Teacher 模型 | 过滤后的样本个数 | 与 k200_val 中样本相似（疑似重复）的样本个数 |
| :-------------: | :------: | :------: | :---------------------: | :--------------: | :------------------------------------------: |
|   k200_train    |  76030   |  45.6G   |           N/A           |       N/A        |                     N/A                      |
|    k200_val     |   4838   |   2.9G   |           N/A           |       N/A        |                     N/A                      |
| googleimage_200 | 3050880  |  265.5G  |      TSN-R50-8seg       |     1188695      |                     967                      |
|  insimage_200   | 3654650  |  224.4G  |      TSN-R50-8seg       |      879726      |                     116                      |
|  insvideo_200   |  732855  | 1487.6G  |    SlowOnly-8x8-R50     |      330680      |                     956                      |
| k200_raw_train  |  76027   |  963.5G  |    SlowOnly-8x8-R50     |       N/A        |                     N/A                      |

MMAction2 所发布的 OmniSource 数据集目录结构如下所示：

```
OmniSource/
├── annotations
│   ├── googleimage_200
│   │   ├── googleimage_200.txt                       从 Google 爬取到的所有图片列表
│   │   ├── tsn_8seg_googleimage_200_duplicate.txt    从 Google 爬取到的，疑似与 k200-val 中样本重复的正样本列表
│   │   ├── tsn_8seg_googleimage_200.txt              从 Google 爬取到的，经过 teacher 模型过滤的正样本列表
│   │   └── tsn_8seg_googleimage_200_wodup.txt        从 Google 爬取到的，经过 teacher 模型过滤及去重的正样本列表
│   ├── insimage_200
│   │   ├── insimage_200.txt
│   │   ├── tsn_8seg_insimage_200_duplicate.txt
│   │   ├── tsn_8seg_insimage_200.txt
│   │   └── tsn_8seg_insimage_200_wodup.txt
│   ├── insvideo_200
│   │   ├── insvideo_200.txt
│   │   ├── slowonly_8x8_insvideo_200_duplicate.txt
│   │   ├── slowonly_8x8_insvideo_200.txt
│   │   └── slowonly_8x8_insvideo_200_wodup.txt
│   ├── k200_actions.txt                              MiniKinetics 中 200 类动作的名称
│   ├── K400_to_MiniKinetics_classidx_mapping.json    Kinetics 中的类索引至 MiniKinetics 中的类索引的映射
│   ├── kinetics_200
│   │   ├── k200_train.txt
│   │   └── k200_val.txt
│   └── kinetics_raw_200
│       └── slowonly_8x8_kinetics_raw_200.json        经 teacher 模型过滤后的 Kinetics 原始视频片段
├── googleimage_200                                   共 10 卷
│   ├── vol_0.tar
│   ├── ...
│   └── vol_9.tar
├── insimage_200                                      共 10 卷
│   ├── vol_0.tar
│   ├── ...
│   └── vol_9.tar
├── insvideo_200                                      共 20 卷
│   ├── vol_00.tar
│   ├── ...
│   └── vol_19.tar
├── kinetics_200_train
│   └── kinetics_200_train.tar
├── kinetics_200_val
│   └── kinetics_200_val.tar
└── kinetics_raw_200_train                            共 16 卷
    ├── vol_0.tar
    ├── ...
    └── vol_15.tar
```

## 数据准备

用户需要首先完成数据下载，对于 `kinetics_200` 和三个网络数据集 `googleimage_200`, `insimage_200`, `insvideo_200`，用户仅需解压各压缩卷并将其合并至一处。

对于 Kinetics 原始视频，由于直接读取长视频非常耗时，用户需要先将其分割为小段。MMAction2 提供了名为 `trim_raw_video.py` 的脚本，用于将长视频分割至 10 秒的小段（分割完成后删除长视频）。用户可利用这一脚本分割长视频。

所有数据应位于 `data/OmniSource/` 目录下。完成数据准备后，`data/OmniSource/` 目录的结构应如下所示（为简洁，省去了训练及测试时未使用的文件）：

```
data/OmniSource/
├── annotations
│   ├── googleimage_200
│   │   └── tsn_8seg_googleimage_200_wodup.txt    Positive file list of images crawled from Google, filtered by the teacher model, after de-duplication.
│   ├── insimage_200
│   │   └── tsn_8seg_insimage_200_wodup.txt
│   ├── insvideo_200
│   │   └── slowonly_8x8_insvideo_200_wodup.txt
│   ├── kinetics_200
│   │   ├── k200_train.txt
│   │   └── k200_val.txt
│   ├── kinetics_raw_200
│   │   └── slowonly_8x8_kinetics_raw_200.json    Kinetics Raw Clips filtered by the teacher model.
│   └── webimage_200
│       └── tsn_8seg_webimage_200_wodup.txt       The union of `tsn_8seg_googleimage_200_wodup.txt` and `tsn_8seg_insimage_200_wodup.txt`
├── googleimage_200
│   ├── 000
|   │   ├── 00
|   │   │   ├── 000001.jpg
|   │   │   ├── ...
|   │   │   └── 000901.jpg
|   │   ├── ...
|   │   ├── 19
│   ├── ...
│   └── 199
├── insimage_200
│   ├── 000
|   │   ├── abseil
|   │   │   ├── 1J9tKWCNgV_0.jpg
|   │   │   ├── ...
|   │   │   └── 1J9tKWCNgV_0.jpg
|   │   ├── abseiling
│   ├── ...
│   └── 199
├── insvideo_200
│   ├── 000
|   │   ├── abseil
|   │   │   ├── B00arxogubl.mp4
|   │   │   ├── ...
|   │   │   └── BzYsP0HIvbt.mp4
|   │   ├── abseiling
│   ├── ...
│   └── 199
├── kinetics_200_train
│   ├── 0074cdXclLU.mp4
|   ├── ...
|   ├── zzzlyL61Fyo.mp4
├── kinetics_200_val
│   ├── 01fAWEHzudA.mp4
|   ├── ...
|   ├── zymA_6jZIz4.mp4
└── kinetics_raw_200_train
│   ├── pref_
│   |   ├── ___dTOdxzXY
|   │   │   ├── part_0.mp4
|   │   │   ├── ...
|   │   │   ├── part_6.mp4
│   |   ├── ...
│   |   └── _zygwGDE2EM
│   ├── ...
│   └── prefZ
```
