# Preparing JHMDB

## Introduction

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

For basic dataset information, you can refer to the dataset [website](http://jhmdb.is.tue.mpg.de/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/jhmdb/`.

## Download and Extract

You can download the RGB frames, optical flow and ground truth annotations from [google drive](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct).
The data are provided from [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md), which is adapted from [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector).

After downloading the `JHMDB.tar.gz` file and put it in `$MMACTION2/tools/data/jhmdb/`, you can run the following command to extract.

```shell
tar -zxvf JHMDB.tar.gz
```

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/JHMDB/
ln -s /mnt/SSD/JHMDB/ ../../../data/jhmdb
```

## Check Directory Structure

After extracting, you will get the `FlowBrox04` directory, `Frames` directory and `JHMDB-GT.pkl` for JHMDB.

In the context of the whole project (for JHMDB only), the folder structure will look like:

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

**Note**: The `JHMDB-GT.pkl` exists as a cache, it contains 6 items as follows:

1. `labels` (list): List of the 21 labels.
2. `gttubes` (dict): Dictionary that contains the ground truth tubes for each video.
  A **gttube** is dictionary that associates with each index of label and a list of tubes.
  A **tube** is a numpy array with `nframes` rows and 5 columns, each col is in format like `<frame index> <x1> <y1> <x2> <y2>`.
3. `nframes` (dict): Dictionary that contains the number of frames for each video, like `'walk/Panic_in_the_Streets_walk_u_cm_np1_ba_med_5': 16`.
4. `train_videos` (list): A list with `nsplits=1` elements, each one containing the list of training videos.
5. `test_videos` (list): A list with `nsplits=1` elements, each one containing the list of testing videos.
6. `resolution` (dict): Dictionary that outputs a tuple (h,w) of the resolution for each video, like `'pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1': (240, 320)`.
