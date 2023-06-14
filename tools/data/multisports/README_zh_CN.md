# 准备 MultiSports

## 介绍

<!-- [DATASET] -->

```BibTeX
@inproceedings{li2021multisports,
  title={Multisports: A multi-person video dataset of spatio-temporally localized sports actions},
  author={Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13536--13545},
  year={2021}
}
```

关于基本数据集信息，请参考官方 [项目](https://deeperaction.github.io/datasets/multisports.html) 和 [论文](https://arxiv.org/abs/2105.07404)。
在我们开始之前，请确保目录位于 `$MMACTION2/tools/data/multisports/`。

## 第一步：准备标注

首先，你必须从官方 [网站](https://github.com/MCG-NJU/MultiSports) 下载标注和视频到 `$MMACTION2/data/multisports`，请同时下载人物检测框并将其放到 `$MMACTION2/data/multisports`。

## 第二步：准备视频

在这一步之前，请确保文件夹结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── multisports
│   |   ├── MultiSports_box.zip
│   |   ├── trainval
│   |   |   ├── aerobic_gymnastics.zip
│   |   |   ├── basketball.zip
│   |   |   ├── multisports_GT.pkl
│   |   |   ├──...
│   |   ├── test
│   |   |   ├── aerobic_gymnastics.zip
│   |   |   ├── basketball.zip
│   |   |   ├──...
```

然后，你可以使用以下命令进行解压。

```shell
cd $MMACTION2/data/multisports/
unzip MultiSports_box.zip
cd $MMACTION2/data/multisports/trainval
find . -name '*.zip' -exec unzip {} \;
cd $MMACTION2/data/multisports/test
find . -name '*.zip' -exec unzip {} \;
cd $MMACTION2/tools/data/multisports/
```

## 第三步：转换标注文件

你可以运行以下脚本来转换我们需要的标注文件和候选框。

```shell
cd $MMACTION2/tools/data/multisports/
python parse_anno.py
```

## 第五步：检查目录结构

完成整个数据处理后，你将得到 MultiSports 数据集的视频和标注文件。

在整个项目的目录中（仅针对 MultiSports），文件夹结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── multisports
│   |   ├── annotations
|   │   |   ├── multisports_dense_proposals_test.recall_96.13.pkl
|   │   |   ├── multisports_dense_proposals_train.recall_96.13.pkl
|   │   |   ├── multisports_dense_proposals_val.recall_96.13.pkl
|   │   |   ├── multisports_GT.pkl
|   │   |   ├── multisports_train.csv
|   │   |   ├── multisports_val.csv
│   |   ├── trainval
│   |   |   ├── aerobic_gymnastics
|   │   |   |   ├── v__wAgwttPYaQ_c001.mp4
|   │   |   |   ├── v__wAgwttPYaQ_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── basketball
|   │   |   |   ├── v_-6Os86HzwCs_c001.mp4
|   │   |   |   ├── v_-6Os86HzwCs_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── multisports_GT.pkl
│   |   |   ├──...
│   |   ├── test
│   |   |   ├── aerobic_gymnastics
|   │   |   |   ├── v_2KroSzspz-c_c001.mp4
|   │   |   |   ├── v_2KroSzspz-c_c002.mp4
|   │   |   |   ├── ...
│   |   |   ├── basketball
|   │   |   |   ├── v_1tefH1iPbGM_c001.mp4
|   │   |   |   ├── v_1tefH1iPbGM_c002.mp4
│   |   |   ├──...
```

我们不需要项目下的 zip 文件，你可以按照自己的意愿处理它们。
关于在 MultiSports 上进行训练和评估，请参考 [训练和测试教程](/docs/en/user_guides/train_test.md)。
