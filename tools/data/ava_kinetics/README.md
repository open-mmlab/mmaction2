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

You need to specify `avakinetics_anotation` which is the directory to ava-kinetics anotations. By default it should be `./ava_kinetics_v1_0`. You need to specify `kinetics_list`, which is the path to the text file containing the paths to all videos as mentioned in Step 0. You need to specify `avakinetics_root`, which is the directory to save the cutted ava-kinetics videos. By default it should be `$MMACTION2/data/ava_kinetics`. 

Here is an example.

```shell
python3 cut_kinetics.py --avakinetics_anotation='./ava_kinetics_v1_0' \
                        --kinetics_list=KINETICS_LIST \
                        --avakinetics_root='../../../data/ava_kinetics' 
```

There should be about 100k videos. It is OK if some videos are missing and we will ignore them in the next steps.


## Step 3. Prepare Annotations

Use `prepare_annotation.py` to prepare the training annotations. It will generate a `kinetics_train.csv` file containning the spatial-temporal annotations for the Kinetics part. This csv file will be written to the same directory as the trimmed video specified by `avakinetics_root`.

You need to specify `avakinetics_anotation` which is the directory to ava-kinetics anotations. By default it should be `./ava_kinetics_v1_0`. You need to specify `avakinetics_root`, which is the directory to save the cutted ava-kinetics videos. By default it should be `$MMACTION2/data/ava_kinetics`. 

Here is an example.

```shell
python3 prepare_annotation.py --avakinetics_anotation='./ava_kinetics_v1_0' \
                              --avakinetics_root='../../../data/ava_kinetics' 
```

## Step 4. Fetch Proposal Files

The pre-computed proposals for AVA dataset are provided by FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks). For the Kinetics part, we use `Cascade R-CNN X-101-64x4d-FPN` from [mmdetection](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth) to get the proposals. You can also use other human detectors to get the pre-computed proposals, and the final action detection performance is similar (since the ground truth bounding bboxes are added to the proposals by default). 

The proposal file is saved at `$MMACTION2/data/ava_kinetics/annotations/kinetics_proposal.pkl`. 
You need to use `--rawframe_path` option to specify the path for rawframe. The default rawframe path is `$MMACTION2/data/ava_kinetics/rawframe/`.      
You can use '--num_gpus' option to specify the number of GPUs used to extract proposals.
```shell
python3 extract_proposal.py --rawframe_path=RAWFRAME_PATH --kinetics_proposal_path=KINETICS_PROPOSAL_PATH
```



## Step 5. Merge AVA and Kinetics 

Now we have finished the preparation of the Kinetics part. We need to merge the AVA part to the ava_kinetics folder (assuming you have already prepared the AVA dataset). Firstly we need to merge the rawframes of AVA part. If you have enough storage, just need to copy (or move) them. 

You can also use soft links:
```shell
python3 softlink_ava.py --absolute_ava_path=ABSOLUTE_AVA_PATH --kinetics_path=KINETICS_PATH
```
The `--absolute_ava_path` option specifies the **absolute path** for the rawframes of AVA dataset.  
The `--kinetics_path` option specifies the path for the rawframes of the Kinetics part.

If the path for the rawframes are not located at $MMACTION2/data/ava_kinetics/rawframe/$, you need to make a soft link to this path.

Next we merge the anotation files.
```shell
cp ../../../data/ava/annotations/* ../../../data/ava_kinetics/annotations/
python3 merge_annotations.py
```

