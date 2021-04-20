# 准备 Charades

## 简介

[DATASET]

```BibTeX
@inproceedings{sigurdsson2016hollywood,
  title={Hollywood in homes: Crowdsourcing data collection for activity understanding},
  author={Sigurdsson, Gunnar A and Varol, G{\"u}l and Wang, Xiaolong and Farhadi, Ali and Laptev, Ivan and Gupta, Abhinav},
  booktitle={European Conference on Computer Vision},
  pages={510--526},
  year={2016},
  organization={Springer}
}
```

用户可参考该数据集的 [官网](https://prior.allenai.org/projects/charades)，以获取数据集相关的基本信息。
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/charades/`。

## 步骤 1. 下载标注文件

首先，用户需要在 [官网](https://prior.allenai.org/projects/charades) 下载标注文件。下载好的标注文件需要放在 `$MMACTION2/data/charades/annotations` 文件夹下。

## 步骤 2. 准备 RGB 帧

用户可在 [官网](https://prior.allenai.org/projects/charades) 直接下载 RGB 帧，并解压至 `$MMACTION2/data/charades/rawframes` 文件夹下。

目前，MMAction2 只支持使用 RGB 帧进行 Charades 数据集的训练和测试，用户也可以在 [官网](https://prior.allenai.org/projects/charades) 直接下载视频和光流到 `$MMACTION2/data/charades/videos` 和 `$MMACTION2/data/charades/rawframes` 文件夹下。

## 步骤 3. 下载文件列表

该文件列表由 [SlowFast](https://github.com/facebookresearch/SlowFast) 生成。

可使用以下脚本下载文件列表用于训练和测试：

```shell
bash fetch_charades_filelist.sh
```

## 步骤 4. 目录结构

在完成所有 Charades 数据集准备流程后，
用户可以获得对应的 RGB 帧文件以及标注文件。

在整个 MMAction2 文件夹下，Charades 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── charades
│   │   ├── annotations
│   |   |   ├── charades_{train,val}_list_rawframes.csv
│   |   |   ├── ...
│   |   ├── rawframes
│   |   |   ├── 001YG
│   |   |   |   ├── 001YG-000001.jpg
│   |   |   |   ├── 001YG-000002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 003WS
│   |   |   ├── ...

```

关于对 Charades 进行训练和验证，可以参考 [基础教程](/docs_zh_CN/getting_started.md)。
