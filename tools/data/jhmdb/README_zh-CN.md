# 准备 JHMDB

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{Jhuang:ICCV:2013,
    title = {Towards understanding action recognition},
    author = {H. Jhuang and J. Gall and S. Zuffi and C. Schmid and M. J. Black},
    booktitle = {International Conf. on Computer Vision (ICCV)},
    month = Dec,
    pages = {3192-3199},
    year = {2013}
}
```

用户可参考该数据集的 [官网](http://jhmdb.is.tue.mpg.de/)，以获取数据集相关的基本信息。
在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/jhmdb/`。

## 下载和解压

用户可以从 [这里](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct) 下载 RGB 帧，光流和真实标签文件。
该数据由 [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md) 代码库提供，参考自 [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector)。

用户在下载 `JHMDB.tar.gz` 文件后，需将其放置在 `$MMACTION2/tools/data/jhmdb/` 目录下，并使用以下指令进行解压：

```shell
tar -zxvf JHMDB.tar.gz
```

如果拥有大量的 SSD 存储空间，则推荐将抽取的帧存储至 I/O 性能更优秀的 SSD 中。

可以运行以下命令为 SSD 建立软链接。

```shell
# 执行这两行进行抽取（假设 SSD 挂载在 "/mnt/SSD/"）
mkdir /mnt/SSD/JHMDB/
ln -s /mnt/SSD/JHMDB/ ../../../data/jhmdb
```

## 检查文件夹结构

完成解压后，用户将得到 `FlowBrox04` 文件夹，`Frames` 文件夹和 `JHMDB-GT.pkl` 文件。

在整个 MMAction2 文件夹下，JHMDB 的文件结构如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── jhmdb
│   |   ├── FlowBrox04
│   |   |   ├── brush_hair
│   |   |   |   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00039.jpg
│   |   |   |   |   ├── 00040.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── Trannydude___Brushing_SyntheticHair___OhNOES!__those_fukin_knots!_brush_hair_u_nm_np1_fr_goo_2
│   |   |   ├── ...
│   |   |   ├── wave
│   |   |   |   ├── 21_wave_u_nm_np1_fr_goo_5
│   |   |   |   ├── ...
│   |   |   |   ├── Wie_man_winkt!!_wave_u_cm_np1_fr_med_0
│   |   ├── Frames
│   |   |   ├── brush_hair
│   |   |   |   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   |   |   |   |   ├── 00001.png
│   |   |   |   |   ├── 00002.png
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00039.png
│   |   |   |   |   ├── 00040.png
│   |   |   |   ├── ...
│   |   |   |   ├── Trannydude___Brushing_SyntheticHair___OhNOES!__those_fukin_knots!_brush_hair_u_nm_np1_fr_goo_2
│   |   |   ├── ...
│   |   |   ├── wave
│   |   |   |   ├── 21_wave_u_nm_np1_fr_goo_5
│   |   |   |   ├── ...
│   |   |   |   ├── Wie_man_winkt!!_wave_u_cm_np1_fr_med_0
│   |   ├── JHMDB-GT.pkl

```

**注意**：`JHMDB-GT.pkl` 作为一个缓存文件，它包含 6 个项目：

1. `labels` (list)：21 个行为类别名称组成的列表
2. `gttubes` (dict)：每个视频对应的基准 tubes 组成的字典
  **gttube** 是由标签索引和 tube 列表组成的字典
  **tube** 是一个 `nframes` 行和 5 列的 numpy array，每一列的形式如 `<frame index> <x1> <y1> <x2> <y2>`
3. `nframes` (dict)：用以表示每个视频对应的帧数，如 `'walk/Panic_in_the_Streets_walk_u_cm_np1_ba_med_5': 16`
4. `train_videos` (list)：包含 `nsplits=1` 的元素，每一项都包含了训练视频的列表
5. `test_videos` (list)：包含 `nsplits=1` 的元素，每一项都包含了测试视频的列表
6. `resolution` (dict)：每个视频对应的分辨率（形如 (h,w)），如 `'pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1': (240, 320)`
