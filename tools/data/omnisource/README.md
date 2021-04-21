# Preparing OmniSource

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{duan2020omni,
  title={Omni-sourced Webly-supervised Learning for Video Recognition},
  author={Duan, Haodong and Zhao, Yue and Xiong, Yuanjun and Liu, Wentao and Lin, Dahua},
  journal={arXiv preprint arXiv:2003.13042},
  year={2020}
}
```

We release a subset of the OmniSource web dataset used in the paper [Omni-sourced Webly-supervised Learning for Video Recognition](https://arxiv.org/abs/2003.13042). Since all web dataset in OmniSource are built based on the Kinetics-400 taxonomy, we select those web data related to the 200 classes in Mini-Kinetics subset (which is proposed in [Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://arxiv.org/pdf/1712.04851.pdf)).

We provide data from all sources that are related to the 200 classes in Mini-Kinetics (including Kinetics trimmed clips, Kinetics untrimmed videos, images from Google and Instagram, video clips from Instagram).  To obtain this dataset, please first fill in the [request form](https://docs.google.com/forms/d/e/1FAIpQLSd8_GlmHzG8FcDbW-OEu__G7qLgOSYZpH-i5vYVJcu7wcb_TQ/viewform?usp=sf_link). We will share the download link to you after your request is received. Since we release all data crawled from the web without any filtering, the dataset is large and it may take some time to download them. We describe the size of the datasets in the following table:

|  Dataset Name   | #samples |  Size   |  Teacher Model   | #samples after filtering | #samples similar to k200_val |
| :-------------: | :------: | :-----: | :--------------: | :----------------------: | :--------------------------: |
|   k200_train    |  76030   |  45.6G  |       N/A        |           N/A            |             N/A              |
|    k200_val     |   4838   |  2.9G   |       N/A        |           N/A            |             N/A              |
| googleimage_200 | 3050880  | 265.5G  |   TSN-R50-8seg   |         1188695          |             967              |
|  insimage_200   | 3654650  | 224.4G  |   TSN-R50-8seg   |          879726          |             116              |
|  insvideo_200   |  732855  | 1487.6G | SlowOnly-8x8-R50 |          330680          |             956              |
| k200_raw_train  |  76027   | 963.5G  | SlowOnly-8x8-R50 |           N/A            |             N/A              |

The file structure of our uploaded OmniSource dataset looks like:

```
OmniSource/
├── annotations
│   ├── googleimage_200
│   │   ├── googleimage_200.txt                       File list of all valid images crawled from Google.
│   │   ├── tsn_8seg_googleimage_200_duplicate.txt    Postive file list of images crawled from Google, which is similar to a validation example.
│   │   ├── tsn_8seg_googleimage_200.txt              Postive file list of images crawled from Google, filtered by the teacher model.
│   │   └── tsn_8seg_googleimage_200_wodup.txt        Postive file list of images crawled from Google, filtered by the teacher model, after de-duplication.
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
│   ├── k200_actions.txt                              The list of action names of the 200 classes in MiniKinetics.
│   ├── K400_to_MiniKinetics_classidx_mapping.json    The index mapping from Kinetics-400 to MiniKinetics.
│   ├── kinetics_200
│   │   ├── k200_train.txt
│   │   └── k200_val.txt
│   ├── kinetics_raw_200
│   │   └── slowonly_8x8_kinetics_raw_200.json        Kinetics Raw Clips filtered by the teacher model.
│   └── webimage_200
│       └── tsn_8seg_webimage_200_wodup.txt           The union of `tsn_8seg_googleimage_200_wodup.txt` and `tsn_8seg_insimage_200_wodup.txt`
├── googleimage_200                                   (10 volumes)
│   ├── vol_0.tar
│   ├── ...
│   └── vol_9.tar
├── insimage_200                                      (10 volumes)
│   ├── vol_0.tar
│   ├── ...
│   └── vol_9.tar
├── insvideo_200                                      (20 volumes)
│   ├── vol_00.tar
│   ├── ...
│   └── vol_19.tar
├── kinetics_200_train
│   └── kinetics_200_train.tar
├── kinetics_200_val
│   └── kinetics_200_val.tar
└── kinetics_raw_200_train                            (16 volumes)
    ├── vol_0.tar
    ├── ...
    └── vol_15.tar
```

## Data Preparation

For data preparation, you need to first download those data. For `kinetics_200` and 3 web datasets: `googleimage_200`, `insimage_200` and `insvideo_200`, you just need to extract each volume and merge their contents.

For Kinetics raw videos, since loading long videos is very heavy, you need to first trim it into clips. Here we provide a script named `trim_raw_video.py`. It trims a long video into 10-second clips and remove the original raw video. You can use it to trim the Kinetics raw video.

The data should be placed in `data/OmniSource/`. When data preparation finished, the folder structure of `data/OmniSource` looks like (We omit the files not needed in training & testing for simplicity):

```
data/OmniSource/
├── annotations
│   ├── googleimage_200
│   │   └── tsn_8seg_googleimage_200_wodup.txt    Postive file list of images crawled from Google, filtered by the teacher model, after de-duplication.
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
