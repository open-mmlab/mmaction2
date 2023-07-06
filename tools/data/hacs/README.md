# Preparing HACS Segments

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{zhao2019hacs,
  title={Hacs: Human action clips and segments dataset for recognition and temporal localization},
  author={Zhao, Hang and Torralba, Antonio and Torresani, Lorenzo and Yan, Zhicheng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8668--8678},
  year={2019}
}
```

### Step 0. Download Videos

Before we start preparing the dataset, please following the official [repository](https://github.com/hangzhaomit/HACS-dataset) to download videos from the HACS Segments dataset. You can submit a request for missing videos to the maintainer of the HACS dataset repository. But you can still prepare the dataset for MMAction2 if some videos are missing.

After you finish downloading the dataset, please move the dataset folder to `$MMACTION2/tools/data/hacs/` or use a soft link. The the folder structure should look like:

```
mmaction2
├── mmaction
├── data
├── configs
├── tools
│   ├── hacs
│   │   ├── slowonly_feature_infer.py
│   │   ├── ..
│   │   ├── data
│   │   │   ├── Applying_sunscreen
│   │   │   │   ├── v_0Ch__DqMPwA.mp4
│   │   │   │   ├── v_9CTDjFHl8WE.mp4
│   │   │   │   ├── ..


```

Before we start, make sure you are at `$MMACTION2/tools/data/hacs/`.

### Step 1. Extract Features

We extract features from the HACS videos using [SlowOnly ResNet50 8x8](/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb.py) pretrained on Kinetics700 dataset. For each video, we uniformly sample 100 video clips and extract the 700-dimensional output (before softmax) as the feature, i.e., the feature shape is 100x700.

First, we generate a video list of the dataset:

```
python generate_list.py
```

It will generate an `hacs_data.txt` file located at `$MMACTION2/tools/data/hacs/` which looks like:

```
Horseback_riding/v_Sr2BSq_8FMw.mp4 0
Horseback_riding/v_EQb6OKoqz3Q.mp4 1
Horseback_riding/v_vYKUV8TRngg.mp4 2
Horseback_riding/v_Y8U0X1F-0ck.mp4 3
Horseback_riding/v_hnspbB7wNh0.mp4 4
Horseback_riding/v_HPhlhrT9IOk.mp4 5
```

Next we use the [slowonly_feature_infer.py](/tools/data/hacs/slowonly_feature_infer.py) config to extract features:

```
# number of GPUs to extract feature
NUM_GPUS=8

# download the pretraining checkpoint
wget https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb_20221013-15b93b10.pth

bash ../mmaction2/tools/dist_test.sh \
    slowonly_feature_infer.py \
    slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb_20221013-15b93b10.pth \
    $NUM_GPUS --dump result.pkl
```

We will get a `result.pkl` that contains the 100x700 feature for each video. We re-write the features into csv format at `$MMACTION2/data/HACS/`:

```
# Make sure you are at $MMACTION2/tools/data/hacs/
python write_feature_csv.py
```

### Step 2. Prepare Annotations

We first download the original annotations from the official repository:

```
wget https://github.com/hangzhaomit/HACS-dataset/raw/master/HACS_v1.1.1.zip
unzip HACS_v1.1.1.zip
```

After unzipping, there should be an `HACS_v1.1.1` folder with an `HACS_segments_v1.1.1.json` file in it.

We generate `hacs_anno_train.json`,  `hacs_anno_val.json` and `hacs_anno_test.json` files at `$MMACTION2/data/HACS/`:

```
python3 generate_anotations.py
```

After the two steps finished, the folder structure of the HACS Segments dataset should look like:

```
mmaction2
├── mmaction
├── data
│   ├── HACS
│   │   ├── hacs_anno_train.json
│   │   ├── hacs_anno_val.json
│   │   ├── hacs_anno_test.json
│   │   ├── slowonly_feature
│   │   │   ├── v_008gY2B8Pf4.csv
│   │   │   ├── v_0095rqic1n8.csv
├── configs
├── tools

```
