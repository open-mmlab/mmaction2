# Preparing MSR-VTT Retrieval/ Video Question-Answering Dataset

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

Before preparing the dataset, please make sure that the directory is located at `$MMACTION2/tools/data/msrvtt/`.

## Step 1. Download Annotation Files

You can directly download the following annotation files related to MSR-VTT from the [Google Drive link](https://drive.google.com/drive/folders/12cr94wT8j7pR09AR2nmQg6o26Y1arI50) provided by [VindLU](https://github.com/klauscc) and place them in the `$MMACTION2/tools/data/msrvtt/annotations` directory:

- [msrvtt_qa_train.json](https://drive.google.com/file/d/12dJq5_7v8FytrJwrPB_f22tET1MmGCNh/view?usp=drive_link)
- [msrvtt_qa_val.json](https://drive.google.com/file/d/138q-A-V8fCC2nBYJgqkQa3gBfXVNbNNd/view?usp=drive_link)
- [msrvtt_qa_test.json](https://drive.google.com/file/d/13IiEcUMHiNppWhGwVY1eAaip6iSJM35A/view?usp=drive_link)
- [msrvtt_qa_answer_list.json](https://drive.google.com/file/d/131euz_dssRkDTk3-ioAS5ZsvIxS_Tt4M/view?usp=drive_link)
- [msrvtt_mc_test.json](https://drive.google.com/file/d/13FrUQ2ZDsNDraP7lfnKvTArPIgdtHuLC/view?usp=drive_link)
- [msrvtt_ret_train9k.json](https://drive.google.com/file/d/13OVo0XRdVWTHlFFxbKg3daYCHsMbJxyd/view?usp=drive_link)
- [msrvtt_ret_train7k.json](https://drive.google.com/file/d/13ID97BX4ExO6mWPIUMp-GzXcPBkviSLx/view?usp=drive_link)
- [msrvtt_ret_test1k.json](https://drive.google.com/file/d/13FLrjI-aleKeU7LbJMDrYgktX7MbTbzu/view?usp=drive_link)
- [msrvtt_test1k.json](https://drive.google.com/file/d/12z6y-DNwIfICSzOhekbJwSbf7z2hlibE/view?usp=drive_link)

## Step 2. Prepare Video Data

You can refer to the [official website](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/) of this dataset for basic information. Run the following commands to prepare the MSRVTT video files:

```shell
# Download original videos
bash download_msrvtt.sh
# Preprocess videos to lower FPS and dimensions
bash compress_msrvtt.sh
```

After completing the above preparation steps, the directory structure will be as follows:

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
