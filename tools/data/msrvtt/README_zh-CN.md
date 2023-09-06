# 准备 MSR-VTT 检索/视频问答数据集

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

在数据集准备前，请确保命令行当前路径为 `$MMACTION2/tools/data/msrvtt/`。

## 步骤 1. 下载标注文件

用户可从 [VindLU](https://github.com/klauscc/VindLU) 提供的 [Google Drive 链接](https://drive.google.com/drive/folders/12cr94wT8j7pR09AR2nmQg6o26Y1arI50)中直接下载以下与 MSR-VTT 相关的标注文件, 并放置到 `$MMACTION2/tools/data/msrvtt/annotations` 路径下:

- [msrvtt_qa_train.json](https://drive.google.com/file/d/12dJq5_7v8FytrJwrPB_f22tET1MmGCNh/view?usp=drive_link)
- [msrvtt_qa_val.json](https://drive.google.com/file/d/138q-A-V8fCC2nBYJgqkQa3gBfXVNbNNd/view?usp=drive_link)
- [msrvtt_qa_test.json](https://drive.google.com/file/d/13IiEcUMHiNppWhGwVY1eAaip6iSJM35A/view?usp=drive_link)
- [msrvtt_qa_answer_list.json](https://drive.google.com/file/d/131euz_dssRkDTk3-ioAS5ZsvIxS_Tt4M/view?usp=drive_link)
- [msrvtt_mc_test.json](https://drive.google.com/file/d/13FrUQ2ZDsNDraP7lfnKvTArPIgdtHuLC/view?usp=drive_link)
- [msrvtt_ret_train9k.json](https://drive.google.com/file/d/13OVo0XRdVWTHlFFxbKg3daYCHsMbJxyd/view?usp=drive_link)
- [msrvtt_ret_train7k.json](https://drive.google.com/file/d/13ID97BX4ExO6mWPIUMp-GzXcPBkviSLx/view?usp=drive_link)
- [msrvtt_ret_test1k.json](https://drive.google.com/file/d/13FLrjI-aleKeU7LbJMDrYgktX7MbTbzu/view?usp=drive_link)
- [msrvtt_test1k.json](https://drive.google.com/file/d/12z6y-DNwIfICSzOhekbJwSbf7z2hlibE/view?usp=drive_link)

## 步骤 2. 准备视频数据

用户可参考该数据集的[官网](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)，以获取数据集相关的基本信息。运行下面的命令准备 MSRVTT 视频文件:

```shell
# download original videos
bash download_msrvtt.sh
# preprocess videos to lower FPS and dimension
bash compress_msrvtt.sh
```

完成上述准备步骤后，文件目录如下：

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   └── msrvtt
│   │   ├── annotations
│   │   │   ├── msrvtt_qa_train.json
│   │   │   ├── msrvtt_qa_val.json
│   │   │   ├── msrvtt_qa_test.json
│   │   │   ├── msrvtt_qa_answer_list.json
│   │   │   ├── msrvtt_mc_test.json
│   │   │   ├── msrvtt_ret_train9k.json
│   │   │   ├── msrvtt_ret_train7k.json
│   │   │   ├── msrvtt_ret_test1k.json
│   │   │   └── msrvtt_test1k.json
│   │   └── videos_2fps_224
│   │       ├── video0.mp4
│   │       ├── video1.mp4
│   │       ├── ...
│   │       └── video9999.mp4
```
