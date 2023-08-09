# 准备 Kinetics-710

## 介绍

<!-- [DATASET] -->

```BibTeX
@misc{li2022uniformerv2,
      title={UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer},
      author={Kunchang Li and Yali Wang and Yinan He and Yizhuo Li and Yi Wang and Limin Wang and Yu Qiao},
      year={2022},
      eprint={2211.09552},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

关于基本数据集信息，请参考 [论文](https://arxiv.org/pdf/2211.09552.pdf)。这些脚本可以用于准备 kinetics-710。MMAction2 以 Concat Daataset 的形式支持了 Kinetics-710 数据集，我们只提供一个注释文件列表，并利用 Kinetics-400/600/700 数据集的原始数据。你可以参考 [配置](/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip_u8_kinetics710-rgb.py) 了解详情，它也提供了一个模板配置，说明了如何在 MMAction2 中使用 Concat Dataset。
在我们开始之前，请确保目录位于 `$MMACTION2`。

## 第一步：下载 Kinetics 400/600/700

Kinetics-710 是基于 Kinetics-400/600/700 的视频数据集，它合并了这些 Kinetics 数据集的训练集，并根据 Youtube ID 删除了重复的视频。MMAction2 提供了一个基于 Kinetics-400/600/700 的 OpenDataLab 版本的标注文件，你可以通过 [MIM](https://github.com/open-mmlab/mim) 从 OpenDataLab 下载。

```shell
# 安装 OpenXLab CLI 工具
pip install -U openxlab
# 登录 OpenXLab
openxlab login
# 下载 Kinetics-400/600/700，注意这可能需要很长时间。
mim download mmaction2 --dataset kinetics400
mim download mmaction2 --dataset kinetics600
mim download mmaction2 --dataset kinetics700

```

## 第二步：下载 Kinetics-710 标注文件

我们提供了与 OpenDataLab 版本 Kinetics 相对应的 Kinetics-710 标注文件列表，你可以从阿里云下载它，并将其解压到 `$MMACTION2/data/`

```shell
wget -P data https://download.openmmlab.com/mmaction/dataset/kinetics710/annotations.zip
cd data && unzip annotations.zip && cd ..

```

## 第三步：文件夹结构

完成 Kinetics 准备的整个数据流程后。
你可以得到 Kinetics-710 的视频和注释文件。

在整个项目目录下（仅针对 Kinetics），*最小*的文件夹结构如下：
（*最小*意味着一些数据是不必要的：例如，你可能想要使用原始视频格式评估 kinetics。）

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── kinetics400
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── jf7RDuUTrsQ.mp4
│   │   │   ├── ...
│   ├── kinetics600
│   │   ├── videos
│   │   │   ├── vol_00
│   │   │   │   ├── -A5JFdMXB_k_000018_000028.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vol63
│   ├── kinetics700
│   │   ├── videos
│   │   │   ├── vol_00
│   │   │   │   ├── -Paa0R0tQ1w_000009_000019.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── vol63
│   ├── kinetics710
│   │   ├── k400_train_list_videos.txt
│   │   ├── k400_val_list_videos.txt
│   │   ├── k600_train_list_videos.txt
│   │   ├── k600_val_list_videos.txt
│   │   ├── k700_train_list_videos.txt
│   │   ├── k700_val_list_videos.txt
```

关于在 Kinetics 上进行训练和评估，请参考 [训练和测试教程](/docs/en/user_guides/train_test.md)。
