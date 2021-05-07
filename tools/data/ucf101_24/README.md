# Preparing UCF101-24

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

For basic dataset information, you can refer to the dataset [website](http://www.thumos.info/download.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ucf101_24/`.

## Download and Extract

You can download the RGB frames, optical flow and ground truth annotations from [google drive](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct).
The data are provided from [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md), which is adapted from [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector) and [corrected-UCF101-Annots](https://github.com/gurkirt/corrected-UCF101-Annots).

**Note**: The annotation of this UCF101-24 is from [here](https://github.com/gurkirt/corrected-UCF101-Annots), which is more correct.

After downloading the `UCF101_v2.tar.gz` file and put it in `$MMACTION2/tools/data/ucf101_24/`, you can run the following command to uncompress.

```shell
tar -zxvf UCF101_v2.tar.gz
```

## Check Directory Structure

After uncompressing, you will get the `rgb-images` directory, `brox-images` directory and `UCF101v2-GT.pkl` for UCF101-24.

In the context of the whole project (for UCF101-24 only), the folder structure will look like:

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

**Note**: The `UCF101v2-GT.pkl` exists as a cache, it contains 6 items as follows:

1. `labels` (list): List of the 24 labels.
2. `gttubes` (dict): Dictionary that contains the ground truth tubes for each video.
  A **gttube** is dictionary that associates with each index of label and a list of tubes.
  A **tube** is a numpy array with `nframes` rows and 5 columns, each col is in format like `<frame index> <x1> <y1> <x2> <y2>`.
3. `nframes` (dict): Dictionary that contains the number of frames for each video, like `'HorseRiding/v_HorseRiding_g05_c02': 151`.
4. `train_videos` (list): A list with `nsplits=1` elements, each one containing the list of training videos.
5. `test_videos` (list): A list with `nsplits=1` elements, each one containing the list of testing videos.
6. `resolution` (dict): Dictionary that outputs a tuple (h,w) of the resolution for each video, like `'FloorGymnastics/v_FloorGymnastics_g09_c03': (240, 320)`.
