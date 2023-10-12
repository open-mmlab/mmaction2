# Preparing AVA

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{gao2017tall,
  title={Tall: Temporal activity localization via language query},
  author={Gao, Jiyang and Sun, Chen and Yang, Zhenheng and Nevatia, Ram},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5267--5275},
  year={2017}
}

@inproceedings{DRN2020CVPR,
  author    = {Runhao, Zeng and Haoming, Xu and Wenbing, Huang and Peihao, Chen and Mingkui, Tan and Chuang Gan},
  title     = {Dense Regression Network for Video Grounding},
  booktitle = {CVPR},
  year      = {2020},
}
```

Charades-STA is a new dataset built on top of Charades by adding sentence temporal annotations. It is introduced by Gao et al. in `TALL: Temporal Activity Localization via Language Query`. Currently, we only support C3D features from `Dense Regression Network for Video Grounding`.

## Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations from the official repository of DRN:

```shell
bash download_annotations.sh
```

## Step 2. Prepare C3D features

After the first step, you should be at `${MMACTION2}/data/CharadesSTA/`. Download the C3D features following the [official command](https://github.com/Alvin-Zeng/DRN/tree/master#download-features) to the current directory `${MMACTION2}/data/CharadesSTA/`.

After finishing the two steps, the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── CharadesSTA
│   │   ├── C3D_unit16_overlap0.5_merged
│   │   |   ├── 001YG.pt
│   │   |   ├── 003WS.pt
│   │   |   ├── 004QE.pt
│   │   |   ├── 00607.pt
│   │   |   ├── ...
│   │   ├── Charades_duration.json
│   │   ├── Charades_fps_dict.json
│   │   ├── Charades_frames_info.json
│   │   ├── Charades_sta_test.txt
│   │   ├── Charades_sta_train.txt
│   │   ├── Charades_word2id.json
```
