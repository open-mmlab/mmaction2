# Preparing Video Retrieval Datasets

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{xu2016msr,
      title={Msr-vtt: A large video description dataset for bridging video and language},
      author={Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
      booktitle={CVPR},
      pages={5288--5296},
      year={2016}
}
```

```BibTeX
@inproceedings{chen2011collecting,
  title={Collecting highly parallel data for paraphrase evaluation},
  author={Chen, David and Dolan, William B},
  booktitle={ACL},
  pages={190--200},
  year={2011}
}
```

Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/video_retrieval/`.

## Preparing MSRVTT dataset

For basic dataset information, you can refer to the MSRVTT dataset [website](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/). Run the following command to prepare the MSRVTT dataset:

```shell
bash prepare_msrvtt.sh
```

After preparation, the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── video_retrieval
│   │   └── msrvtt
│   │       ├── train_9k.json
│   │       ├── train_7k.json
│   │       ├── test_JSFUSION.json
│   │       └─── videos
│   │           ├── video0.mp4
│   │           ├── video1.mp4
│   │           ├── ...
│   │           └── video9999.mp4
```

## Preparing MSVD dataset

For basic dataset information, you can refer to the MSVD dataset [website](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/). Run the following command to prepare the MSVD dataset:

```shell
bash prepare_msvd.sh
```

After preparation, the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── video_retrieval
│   │   └── msrvd
│   │       ├── train.json
│   │       ├── test.json
│   │       ├── val.json
│   │       └─── videos
│   │           ├── xxx.avi
│   │           ├── xxx.avi
│   │           ├── ...
│   │           └── xxx.avi
```
