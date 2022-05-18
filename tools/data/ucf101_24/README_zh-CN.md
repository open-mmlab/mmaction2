# 准备 UCF101-24

## 简介

```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

用户可参考该数据集的 [官网](http://www.thumos.info/download.html)，以获取数据集相关的基本信息。
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/ucf101_24/`。

## 下载和解压

用户可以从 [这里](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct) 下载 RGB 帧，光流和标注文件。
该数据由 [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md) 代码库提供，
参考自 [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector) 和 [corrected-UCF101-Annots](https://github.com/gurkirt/corrected-UCF101-Annots)。

**注意**：UCF101-24 的标注文件来自于 [这里](https://github.com/gurkirt/corrected-UCF101-Annots)，该标注文件相对于其他标注文件更加准确。

用户在下载 `UCF101_v2.tar.gz` 文件后，需将其放置在 `$MMACTION2/tools/data/ucf101_24/` 目录下，并使用以下指令进行解压：

```shell
tar -zxvf UCF101_v2.tar.gz
```

## 检查文件夹结构

经过解压后，用户将得到 `rgb-images` 文件夹，`brox-images` 文件夹和 `UCF101v2-GT.pkl` 文件。

在整个 MMAction2 文件夹下，UCF101_24 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101_24
│   |   ├── brox-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── rgb-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── UCF101v2-GT.pkl

```

**注意**：`UCF101v2-GT.pkl` 作为一个缓存文件，它包含 6 个项目：

1. `labels` (list)：24 个行为类别名称组成的列表
2. `gttubes` (dict)：每个视频对应的基准 tubes 组成的字典
   **gttube** 是由标签索引和 tube 列表组成的字典
   **tube** 是一个 `nframes` 行和 5 列的 numpy array，每一列的形式如 `<frame index> <x1> <y1> <x2> <y2>`
3. `nframes` (dict)：用以表示每个视频对应的帧数，如 `'HorseRiding/v_HorseRiding_g05_c02': 151`
4. `train_videos` (list)：包含 `nsplits=1` 的元素，每一项都包含了训练视频的列表
5. `test_videos` (list)：包含 `nsplits=1` 的元素，每一项都包含了测试视频的列表
6. `resolution` (dict)：每个视频对应的分辨率（形如 (h,w)），如 `'FloorGymnastics/v_FloorGymnastics_g09_c03': (240, 320)`
