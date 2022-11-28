# Preparing AVA-Kinetics

## Introduction

<!-- [DATASET] -->

```BibTeX
@article{li2020ava,
  title={The ava-kinetics localized human actions video dataset},
  author={Li, Ang and Thotakuri, Meghana and Ross, David A and Carreira, Jo{\~a}o and Vostrikov, Alexander and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2005.00214},
  year={2020}
}
```

For basic dataset information, please refer to the official [website](https://research.google.com/ava/index.html).
AVA-Kinetics dataset is a crossover between the AVA Actions and Kinetics datasets. You may want to first prepare the AVA datasets. In this file, we provide commands to prepare the Kinetics part and merge the two parts together.

For model training, we will keep reading from raw frames for the AVA part, but read from videos using `decord` for the Kinetics part to accelerate training.

Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ava_kinetics/`.

## Step 0. Prepare Kinetics700 dataset

The Kinetics part of the AVA-Kinetics dataset comes from Kinetics-700 videos. Best if you already have this dataset locally (only video is needed). If not, see [preparing_kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics). You can just download part of the Kinetics-700 dataset if you do not have enough storage space.

After finishing downloading the videos, please generate a text file showing the paths to all videos:

```
Path_to_video1\n
Path_to_video2\n
...
Path_to_videon\n
```

The timestamp (start and end of the video) must be contained in the file name. For example:

```
class602/o3lCwWyyc_s_000012_000022.mp4\n
```

It means that this video clip is from the 12th to the 22nd second of the original video. It is OK if some videos are missing and we will ignore them in the next steps.

## Step 1. Download Annotations

Download the annotation tar file (the directory should be located at `$MMACTION2/tools/data/ava_kinetics/`).

```shell
wget https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0.tar.gz
tar xf ava_kinetics_v1_0.tar.gz && rm ava_kinetics_v1_0.tar.gz
```

You should have the `ava_kinetics_v1_0` folder at `$MMACTION2/tools/data/ava_kinetics/`.

## Step 2. Cut Videos

Use `cut_kinetics.py` to find the desired videos from the Kinetics-700 dataset and trim them to contain only annotated clips. Currently we only use the train set of the Kinetics part to improve training. Validation on the Kinetics part will come soon.

You need to specify `avakinetics_anotation` which is the directory to ava-kinetics anotations. By default it should be `./ava_kinetics_v1_0`. You need to specify `kinetics_list`, which is the path to the text file containing the paths to all videos as mentioned in Step 0. You need to specify `avakinetics_root`, which is the directory to save the trimmed ava-kinetics videos. By default it should be `$MMACTION2/data/ava_kinetics`.

Here is an example.

```shell
python3 cut_kinetics.py --avakinetics_anotation='./ava_kinetics_v1_0' \
                        --kinetics_list=KINETICS_LIST \
                        --avakinetics_root='../../../data/ava_kinetics'
```

There should be about 100k videos. It is OK if some videos are missing and we will ignore them in the next steps.

## Step 3. Prepare Annotations

Use `prepare_annotation.py` to prepare the training annotations. It will generate a `kinetics_train.csv` file containning the spatial-temporal annotations for the Kinetics part. This csv file will be written to the same directory as the trimmed video specified by `avakinetics_root`.

You need to specify `avakinetics_anotation` which is the directory to ava-kinetics anotations. By default it should be `./ava_kinetics_v1_0`. You need to specify `avakinetics_root`, which is the directory to save the trimmed ava-kinetics videos. By default it should be `$MMACTION2/data/ava_kinetics`.

Here is an example.

```shell
python3 prepare_annotation.py --avakinetics_anotation='./ava_kinetics_v1_0' \
                              --avakinetics_root='../../../data/ava_kinetics'
```

## Step 4. Fetch Proposal Files

The pre-computed proposals for AVA dataset are provided by FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks). For the Kinetics part, we use `Cascade R-CNN X-101-64x4d-FPN` from [mmdetection](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth) to fetch the proposals. Here is the script:

```shell
python3 fetch_proposal.py --datalist='../../../data/ava_kinetics/kinetics_train.csv' \
    --picklepath='../../../data/ava_kinetics/kinetics_proposal.pkl'
```

## Step 5. Merge AVA and Kinetics
