# 准备视频检索数据集

## 简介

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

在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/video_retrieval/`。

## 准备 MSRVTT 数据集

用户可参考该数据集的[官网](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)，以获取数据集相关的基本信息。运行下面的命令准备 MSRVTT 数据集：

```shell
bash prepare_msrvtt.sh
```

完成上述准备步骤后，文件目录如下：

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

## 准备 MSVD 数据集

用户可参考该数据集的[官网](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)，以获取数据集相关的基本信息。运行下面的命令准备 MSVD 数据集：

```shell
bash prepare_msvd.sh
```

完场上述准备步骤后，文件目录如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── video_retrieval
│   │   └── msvd
│   │       ├── train.json
│   │       ├── text.json
│   │       ├── val.json
│   │       └─── videos
│   │           ├── xxx.avi
│   │           ├── xxx.avi
│   │           ├── ...
│   │           └── xxx.avi
```
