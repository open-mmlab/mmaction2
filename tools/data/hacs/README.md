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

Before we start preparing the dataset, please following the offical [repository](https://github.com/hangzhaomit/HACS-dataset) to download videos from the HACS Segments dataset. You can submit a request for missing videos to the maintainer of the HACS dataset repository. But you can still prepare the dataset for MMAction2 if some videos are missing.

After you finish downloading the dataset, please move the dataset folder to `$MMACTION2/tools/data/hacs/` or use a soft link. The the folder structure should look like:

```
mmaction2
├── mmaction
├── tools
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

### Step 1. Extract Features

We extract features from the HACS videos using [SlowOnly ResNet50 8x8](/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb.py) pretrained on Kinetics700 dataset. We uniformly sample 100 video clips and extract the 700-dimensional output (before softmax) as the feature for each video, i.e., the feature shape is 100x700.

First, we generate a video list of the dataset:
```
python generate_list.py
```

It will generate a `hacs_data.txt` file located at `$MMACTION2/tools/data/hacs/` which looks like:
```
Horseback_riding/v_Sr2BSq_8FMw.mp4 -1
Horseback_riding/v_EQb6OKoqz3Q.mp4 -1
Horseback_riding/v_vYKUV8TRngg.mp4 -1
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