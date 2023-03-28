## ActivityNet

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{Heilbron2015ActivityNetAL,
  title={ActivityNet: A large-scale video benchmark for human activity understanding},
  author={Fabian Caba Heilbron and Victor Escorcia and Bernard Ghanem and Juan Carlos Niebles},
  journal={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={961-970}
}
```

For basic dataset information, please refer to the official [website](http://activity-net.org/).
For action detection, you can either use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network##code-and-data-preparation) or extract feature with mmaction2 (which has better performance).
We release both pipeline.
Before we start, please make sure that current working directory is `$MMACTION2/tools/data/activitynet/`.

### Option 1: Use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation)

#### Step 1. Download Annotations

First of all, you can run the following script to download annotation files.

```shell
bash download_feature_annotations.sh
```

#### Step 2. Prepare Videos Features

Then, you can run the following script to download activitynet features.

```shell
bash download_features.sh
```

#### Step 3. Process Annotation Files

Next, you can run the following script to process the downloaded annotation files for training and testing.
It first merges the two annotation files together and then separates the annoations by `train`, `val` and `test`.

```shell
python process_annotations.py
```

### Option 2: Extract ActivityNet feature using MMAction2 with all videos provided in official [website](http://activity-net.org/)

#### Step 1. Download Annotations

First of all, you can run the following script to download annotation files.

```shell
bash download_annotations.sh
```

#### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

Since some videos in the ActivityNet dataset might be no longer available on YouTube, official [website](http://activity-net.org/) has made the full dataset available on Google and Baidu drives.
To accommodate missing data requests, you can fill in this [request form](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) provided in official [download page](http://activity-net.org/download.html) to have a 7-day-access to download the videos from the drive folders.

We also provide download steps for annotations from [BSN repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network##code-and-data-preparation)

```shell
bash download_bsn_videos.sh
```

For this case, the downloading scripts update the annotation file after downloading to make sure every video in it exists.

#### Step 3. Extract RGB and Flow

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Use following scripts to extract both RGB and Flow.

```shell
bash extract_frames.sh
```

The command above can generate images with new short edge 256. If you want to generate images with short edge 320 (320p), or with fix size 340x256, you can change the args `--new-short 256` to `--new-short 320` or `--new-width 340 --new-height 256`.
More details can be found in [data_preparation](/docs/data_preparation.md)

#### Step 4. Generate File List for ActivityNet Finetuning

With extracted frames, you can generate video-level or clip-level lists of rawframes, which can be used for ActivityNet Finetuning.

```shell
python generate_rawframes_filelist.py
```

#### Step 5. Finetune TSN models on ActivityNet

You can use ActivityNet configs in `configs/recognition/tsn` to finetune TSN models on ActivityNet.
You need to use Kinetics models for pretraining.
Both RGB models and Flow models are supported.

#### Step 6. Extract ActivityNet Feature with finetuned ckpts

After finetuning TSN on ActivityNet, you can use it to extract both RGB and Flow feature.

```shell
python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/rgb_feat --modality RGB --ckpt /path/to/rgb_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_train_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth

python tsn_feature_extraction.py --data-prefix ../../../data/ActivityNet/rawframes --data-list ../../../data/ActivityNet/anet_val_video.txt --output-prefix ../../../data/ActivityNet/flow_feat --modality Flow --ckpt /path/to/flow_checkpoint.pth
```

After feature extraction, you can use our post processing scripts to concat RGB and Flow feature, generate the `100-t X 400-d` feature for Action Detection.

```shell
python activitynet_feature_postprocessing.py --rgb ../../../data/ActivityNet/rgb_feat --flow ../../../data/ActivityNet/flow_feat --dest ../../../data/ActivityNet/mmaction_feat
```

### Final Step. Check Directory Structure

After the whole data pipeline for ActivityNet preparation,
you will get the features, videos, frames and annotation files.

In the context of the whole project (for ActivityNet only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet

(if Option 1 used)
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..

(if Option 2 used)
│   │   ├── anet_train_video.txt
│   │   ├── anet_val_video.txt
│   │   ├── anet_train_clip.txt
│   │   ├── anet_val_clip.txt
│   │   ├── activity_net.v1-3.min.json
│   │   ├── mmaction_feat
│   │   │   ├── v___c8enCfzqw.csv
│   │   │   ├── v___dXUJsj3yo.csv
│   │   │   ├── ..
│   │   ├── rawframes
│   │   │   ├── v___c8enCfzqw
│   │   │   │   ├── img_00000.jpg
│   │   │   │   ├── flow_x_00000.jpg
│   │   │   │   ├── flow_y_00000.jpg
│   │   │   │   ├── ..
│   │   │   ├── ..

```

For training and evaluating on ActivityNet, please refer to [getting_started.md](/docs/getting_started.md).

## AVA-Kinetics

### Introduction

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

### Step 1. Prepare the Kinetics700 dataset

The Kinetics part of the AVA-Kinetics dataset are sampled from the Kinetics-700 dataset.

It is best if you have prepared the Kinetics-700 dataset (only videos required) following
[Preparing Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics). We will also have alternative method to prepare these videos if you do not have enough storage (coming soon).

We will need the videos of this dataset (`$MMACTION2/data/kinetics700/videos_train`) and the videos file list (`$MMACTION2/data/kinetics700/kinetics700_train_list_videos.txt`), which is generated by [Step 4 in Preparing Kinetics](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/kinetics##step-4-generate-file-list)

The format of the file list should be:

```
Path_to_video_1 label_1\n
Path_to_video_2 label_2\n
...
Path_to_video_n label_n\n
```

The timestamp (start and end of the video) must be contained. For example:

```
class602/o3lCwWyyc_s_000012_000022.mp4 602\n
```

It means that this video clip is the 12th to 22nd seconds of the original video. It is okay if some videos are missing, and we will ignore them in the next steps.

### Step 2. Download Annotations

Download the annotation tar file (recall that the directory should be located at `$MMACTION2/tools/data/ava_kinetics/`).

```shell
wget https://storage.googleapis.com/deepmind-media/Datasets/ava_kinetics_v1_0.tar.gz
tar xf ava_kinetics_v1_0.tar.gz && rm ava_kinetics_v1_0.tar.gz
```

You should have the `ava_kinetics_v1_0` folder at `$MMACTION2/tools/data/ava_kinetics/`.

### Step 3. Cut Videos

Use `cut_kinetics.py` to find the desired videos from the Kinetics-700 dataset and trim them to contain only annotated clips. Currently we only use the train set of the Kinetics part to improve training. Validation on the Kinetics part will come soon.

Here is the script:

```shell
python3 cut_kinetics.py --avakinetics_anotation=$AVAKINETICS_ANOTATION \
                        --kinetics_list=$KINETICS_LIST \
                        --avakinetics_root=$AVAKINETICS_ROOT \
                        [--num_workers=$NUM_WORKERS ]
```

Arguments:

- `avakinetics_anotation`: the directory to ava-kinetics anotations. Defaults to `./ava_kinetics_v1_0`.
- `kinetics_list`: the path to the videos file list as mentioned in Step 1. If you have prepared the Kinetics700 dataset following `mmaction2`, it should be `$MMACTION2/data/kinetics700/kinetics700_train_list_videos.txt`.
- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to cut videos. Defaults to -1 and use all available cpus.

There should be about 100k videos. It is OK if some videos are missing and we will ignore them in the next steps.

### Step 4. Extract RGB Frames

This step is similar to Step 4 in [Preparing AVA](https://github.com/open-mmlab/mmaction2/tree/dev-1.x/tools/data/ava##step-4-extract-rgb-and-flow).

Here we provide a script to extract RGB frames using ffmpeg:

```shell
python3 extract_rgb_frames.py --avakinetics_root=$AVAKINETICS_ROOT \
                              [--num_workers=$NUM_WORKERS ]
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to extract frames. Defaults to -1 and use all available cpus.

If you have installed denseflow, you can also use `build_rawframes.py` to extract RGB frames:

```shell
python3 ../build_rawframes.py ../../../data/ava_kinetics/videos/ ../../../data/ava_kinetics/rawframes/ --task rgb --level 1 --mixed-ext
```

### Step 5. Prepare Annotations

Use `prepare_annotation.py` to prepare the training annotations. It will generate a `kinetics_train.csv` file containning the spatial-temporal annotations for the Kinetics part, localting at `$AVAKINETICS_ROOT`.

Here is the script:

```shell
python3 prepare_annotation.py --avakinetics_anotation=$AVAKINETICS_ANOTATION \
                              --avakinetics_root=$AVAKINETICS_ROOT \
                              [--num_workers=$NUM_WORKERS]
```

Arguments:

- `avakinetics_anotation`: the directory to ava-kinetics anotations. Defaults to `./ava_kinetics_v1_0`.
- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `num_workers`: number of workers used to prepare annotations. Defaults to -1 and use all available cpus.

### Step 6. Fetch Proposal Files

The pre-computed proposals for AVA dataset are provided by FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks). For the Kinetics part, we use `Cascade R-CNN X-101-64x4d-FPN` from [mmdetection](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth) to fetch the proposals. Here is the script:

```shell
python3 fetch_proposal.py --avakinetics_root=$AVAKINETICS_ROOT \
                          --datalist=$DATALIST \
                          --picklepath=$PICKLEPATH \
                          [--config=$CONFIG ] \
                          [--checkpoint=$CHECKPOINT ]

```

It  will generate a `kinetics_proposal.pkl` file at `$MMACTION2/data/ava_kinetics/`.

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `datalist`: path to the `kinetics_train.csv` file generated at Step 3.
- `picklepath`: path to save the extracted proposal pickle file.
- `config`: the config file for the human detection model. Defaults to `X-101-64x4d-FPN.py`.
- `checkpoint`: the checkpoint for the human detection model. Defaults to the `mmdetection` pretraining checkpoint.

### Step 7. Merge AVA to AVA-Kinetics

Now we are done with the preparations for the Kinetics part. We need to merge the AVA part into the `ava_kinetics` folder (assuming you have AVA dataset ready at `$MMACTION2/data/ava`). First we make a copy of the AVA anotation to the `ava_kinetics` folder (recall that you are at `$MMACTION2/tools/data/ava_kinetics/`):

```shell
cp -r ../../../data/ava/annotations/ ../../../data/ava_kinetics/
```

Next we merge the generated anotation files of the Kinetics part to AVA. Please check: you should have two files `kinetics_train.csv` and `kinetics_proposal.pkl` at `$MMACTION2/data/ava_kinetics/` generated from Step 5 and Step 6. Run the following script to merge these two files into `$MMACTION2/data/ava_kinetics/annotations/ava_train_v2.2.csv` and `$MMACTION2/data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl` respectively.

```shell
python3 merge_annotations.py --avakinetics_root=$AVAKINETICS_ROOT
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.

Finally, we need to merge the rawframes of AVA part. You can either copy/move them or generate soft links. The following script is an example to use soft links:

```shell
python3 softlink_ava.py --avakinetics_root=$AVAKINETICS_ROOT \
                        --ava_root=$AVA_ROOT
```

Arguments:

- `avakinetics_root`: the directory to save the ava-kinetics dataset. Defaults to `$MMACTION2/data/ava_kinetics`.
- `ava_root`: the directory to save the ava dataset. Defaults to `$MMACTION2/data/ava`.

## AVA

### Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and Pantofaru, Caroline and Li, Yeqing and Vijayanarasimhan, Sudheendra and Toderici, George and Ricco, Susanna and Sukthankar, Rahul and others},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6047--6056},
  year={2018}
}
```

For basic dataset information, please refer to the official [website](https://research.google.com/ava/index.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ava/`.

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

This command will download `ava_v2.1.zip` for AVA `v2.1` annotation. If you need the AVA `v2.2` annotation, you can try the following script.

```shell
VERSION=2.2 bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, use the following script to prepare videos. The codes are adapted from the [official crawler](https://github.com/cvdfoundation/ava-dataset).
Note that this might take a long time.

```shell
bash download_videos.sh
```

Or you can use the following command to downloading AVA videos in parallel using a python script.

```shell
bash download_videos_parallel.sh
```

Note that if you happen to have sudoer or have [GNU parallel](https://www.gnu.org/software/parallel/) on your machine,
you can speed up the procedure by downloading in parallel.

```shell
## sudo apt-get install parallel
bash download_videos_gnu_parallel.sh
```

### Step 3. Cut Videos

Cut each video from its 15th to 30th minute and make them at 30 fps.

```shell
bash cut_videos.sh
```

### Step 4. Extract RGB and Flow

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ava_extracted/
ln -s /mnt/SSD/ava_extracted/ ../data/ava/rawframes/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using ffmpeg by the following script.

```shell
bash extract_rgb_frames_ffmpeg.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

### Step 5. Fetch Proposal Files

The scripts are adapted from FAIR's [Long-Term Feature Banks](https://github.com/facebookresearch/video-long-term-feature-banks).

Run the following scripts to fetch the pre-computed proposal list.

```shell
bash fetch_ava_proposals.sh
```

### Step 6. Folder Structure

After the whole data pipeline for AVA preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for AVA.

In the context of the whole project (for AVA only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate AVA using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ava
│   │   ├── annotations
│   │   |   ├── ava_dense_proposals_train.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_val.FAIR.recall_93.9.pkl
│   │   |   ├── ava_dense_proposals_test.FAIR.recall_93.9.pkl
│   │   |   ├── ava_train_v2.1.csv
│   │   |   ├── ava_val_v2.1.csv
│   │   |   ├── ava_train_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_val_excluded_timestamps_v2.1.csv
│   │   |   ├── ava_action_list_v2.1_for_activitynet_2018.pbtxt
│   │   ├── videos
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── videos_15min
│   │   │   ├── 053oq2xB3oU.mkv
│   │   │   ├── 0f39OWEqJ24.mp4
│   │   │   ├── ...
│   │   ├── rawframes
│   │   │   ├── 053oq2xB3oU
|   │   │   │   ├── img_00001.jpg
|   │   │   │   ├── img_00002.jpg
|   │   │   │   ├── ...
```

For training and evaluating on AVA, please refer to [getting_started](/docs/getting_started.md).

### Reference

1. O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014

## Diving48

### Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{li2018resound,
  title={Resound: Towards action recognition without representation bias},
  author={Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={513--528},
  year={2018}
}
```

For basic dataset information, you can refer to the official dataset [website](http://www.svcl.ucsd.edu/projects/resound/dataset.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/diving48/`.

### Step 1. Prepare Annotations

You can run the following script to download annotations (considering the correctness of annotation files, we only download V2 version here).

```shell
bash download_annotations.sh
```

### Step 2. Prepare Videos

You can run the following script to download videos.

```shell
bash download_videos.sh
```

### Step 3. Prepare RGB and Flow

This part is **optional** if you only want to use the video loader.

The frames provided in official compressed file are not complete. You may need to go through the following extraction steps to get the complete frames.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/diving48_extracted/
ln -s /mnt/SSD/diving48_extracted/ ../../../data/diving48/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/diving48/
bash extract_frames.sh
```

### Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for Diving48 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Diving48.

In the context of the whole project (for Diving48 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── diving48
│   │   ├── diving48_{train,val}_list_rawframes.txt
│   │   ├── diving48_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   |   ├── Diving48_V2_train.json
│   |   |   ├── Diving48_V2_test.json
│   |   |   ├── Diving48_vocab.json
│   |   ├── videos
│   |   |   ├── _8Vy3dlHg2w_00000.mp4
│   |   |   ├── _8Vy3dlHg2w_00001.mp4
│   |   |   ├── ...
│   |   ├── rawframes
│   |   |   ├── 2x00lRzlTVQ_00000
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2x00lRzlTVQ_00001
│   |   |   ├── ...
```

For training and evaluating on Diving48, please refer to [getting_started.md](/docs/getting_started.md).

## GYM

### Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{shao2020finegym,
  title={Finegym: A hierarchical video dataset for fine-grained action understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2616--2625},
  year={2020}
}
```

For basic dataset information, please refer to the official [project](https://sdolivia.github.io/FineGym/) and the [paper](https://arxiv.org/abs/2004.06704).
We currently provide the data pre-processing pipeline for GYM99.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/gym/`.

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

### Step 3. Trim Videos into Events

First, you need to trim long videos into events based on the annotation of GYM with the following scripts.

```shell
python trim_event.py
```

### Step 4. Trim Events into Subactions

Then, you need to trim events into subactions based on the annotation of GYM with the following scripts. We use the two stage trimming for better efficiency (trimming multiple short clips from a long video can be extremely inefficient, since you need to go over the video many times).

```shell
python trim_subaction.py
```

### Step 5. Extract RGB and Flow

This part is **optional** if you only want to use the video loader for RGB model training.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

Run the following script to extract both rgb and flow using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

### Step 6. Generate file list for GYM99 based on extracted subactions

You can use the following script to generate train / val lists for GYM99.

```shell
python generate_file_list.py
```

### Step 7. Folder Structure

After the whole data pipeline for GYM preparation. You can get the subaction clips, event clips, raw videos and GYM99 train/val lists.

In the context of the whole project (for GYM only), the full folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── gym
|   |   ├── annotations
|   |   |   ├── gym99_train_org.txt
|   |   |   ├── gym99_val_org.txt
|   |   |   ├── gym99_train.txt
|   |   |   ├── gym99_val.txt
|   |   |   ├── annotation.json
|   |   |   └── event_annotation.json
│   │   ├── videos
|   |   |   ├── 0LtLS9wROrk.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw.mp4
│   │   ├── events
|   |   |   ├── 0LtLS9wROrk_E_002407_002435.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006732_006824.mp4
│   │   ├── subactions
|   |   |   ├── 0LtLS9wROrk_E_002407_002435_A_0003_0005.mp4
|   |   |   ├── ...
|   |   |   └── zfqS-wCJSsw_E_006244_006252_A_0000_0007.mp4
|   |   └── subaction_frames
```

For training and evaluating on GYM, please refer to [getting_started](/docs/getting_started.md).

## HMDB51

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{Kuehne2011HMDBAL,
  title={HMDB: A large video database for human motion recognition},
  author={Hilde Kuehne and Hueihan Jhuang and E. Garrote and T. Poggio and Thomas Serre},
  journal={2011 International Conference on Computer Vision},
  year={2011},
  pages={2556-2563}
}
```

For basic dataset information, you can refer to the dataset [website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/hmdb51/`.

To run the bash scripts below, you need to install `unrar`. you can install it by `sudo apt-get install unrar`,
or refer to [this repo](https://github.com/innerlee/setup) by following the usage and taking [`zzunrar.sh`](https://github.com/innerlee/setup/blob/master/zzunrar.sh)
script for easy installation without sudo.

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
bash download_videos.sh
```

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/hmdb51_extracted/
ln -s /mnt/SSD/hmdb51_extracted/ ../../../data/hmdb51/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

### Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for HMDB51 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for HMDB51.

In the context of the whole project (for HMDB51 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hmdb51
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── hmdb51_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi

│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0.avi
│   │   ├── rawframes
│   │   │   ├── brush_hair
│   │   │   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── wave
│   │   │   │   ├── 20060723sfjffbartsinger_wave_f_cm_np1_ba_med_0
│   │   │   │   ├── ...
│   │   │   │   ├── winKen_wave_u_cm_np1_ri_bad_1

```

For training and evaluating on HMDB51, please refer to [getting_started.md](/docs/getting_started.md).

## HVU

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{Diba2019LargeSH,
  title={Large Scale Holistic Video Understanding},
  author={Ali Diba and M. Fayyaz and Vivek Sharma and Manohar Paluri and Jurgen Gall and R. Stiefelhagen and L. Gool},
  journal={arXiv: Computer Vision and Pattern Recognition},
  year={2019}
}
```

For basic dataset information, please refer to the official [project](https://github.com/holistic-video-understanding/HVU-Dataset/) and the [paper](https://arxiv.org/abs/1904.11451).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/hvu/`.

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

Besides, you need to run the following command to parse the tag list of HVU.

```shell
python parse_tag_list.py
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh
```

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

You can use the following script to extract both RGB and Flow frames.

```shell
bash extract_frames.sh
```

By default, we generate frames with short edge resized to 256.
More details can be found in [data_preparation](/docs/data_preparation.md)

### Step 4. Generate File List

You can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh
## execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh
```

### Step 5. Generate File List for Each Individual Tag Categories

This part is **optional** if you don't want to train models on HVU for a specific tag category.

The file list generated in step 4 contains labels of different categories. These file lists can only be
handled with HVUDataset and used for multi-task learning of different tag categories. The component
`LoadHVULabel` is needed to load the multi-category tags, and the `HVULoss` should be used to train
the model.

If you only want to train video recognition models for a specific tag category, i.e. you want to train
a recognition model on HVU which only handles tags in the category `action`, we recommend you to use
the following command to generate file lists for the specific tag category. The new list, which only
contains tags of a specific category, can be handled with `VideoDataset` or `RawframeDataset`. The
recognition models can be trained with `BCELossWithLogits`.

The following command generates file list for the tag category ${category}, note that the tag category you
specified should be in the 6 tag categories available in HVU: \['action', 'attribute', 'concept', 'event',
'object', 'scene'\].

```shell
python generate_sub_file_list.py path/to/filelist.json ${category}
```

The filename of the generated file list for ${category} is generated by replacing `hvu` in the original
filename with `hvu_${category}`. For example, if the original filename is `hvu_train.json`, the filename
of the file list for action is `hvu_action_train.json`.

### Step 6. Folder Structure

After the whole data pipeline for HVU preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for HVU.

In the context of the whole project (for HVU only), the full folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── hvu
│   │   ├── hvu_train_video.json
│   │   ├── hvu_val_video.json
│   │   ├── hvu_train.json
│   │   ├── hvu_val.json
│   │   ├── annotations
│   │   ├── videos_train
│   │   │   ├── OLpWTpTC4P8_000570_000670.mp4
│   │   │   ├── xsPKW4tZZBc_002330_002430.mp4
│   │   │   ├── ...
│   │   ├── videos_val
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

For training and evaluating on HVU, please refer to [getting_started](/docs/getting_started.md).

## Jester

### Introduction

<!-- [DATASET] -->

```BibTeX
@InProceedings{Materzynska_2019_ICCV,
  author = {Materzynska, Joanna and Berger, Guillaume and Bax, Ingo and Memisevic, Roland},
  title = {The Jester Dataset: A Large-Scale Video Dataset of Human Gestures},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2019}
}
```

For basic dataset information, you can refer to the dataset [website](https://developer.qualcomm.com/software/ai-datasets/jester).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/jester/`.

### Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION2/data/jester/annotations` on the official [website](https://developer.qualcomm.com/software/ai-datasets/jester).

### Step 2. Prepare RGB Frames

Since the [jester website](https://developer.qualcomm.com/software/ai-datasets/jester) doesn't provide the original video data and only extracted RGB frames are available, you have to directly download RGB frames from [jester website](https://developer.qualcomm.com/software/ai-datasets/jester).

You can download all RGB frame parts on [jester website](https://developer.qualcomm.com/software/ai-datasets/jester) to `$MMACTION2/data/jester/` and use the following command to extract.

```shell
cd $MMACTION2/data/jester/
cat 20bn-jester-v1-?? | tar zx
cd $MMACTION2/tools/data/jester/
```

For users who only want to use RGB frames, you can skip to step 5 to generate file lists in the format of rawframes. Since the prefix of official JPGs is "%05d.jpg" (e.g., "00001.jpg"),
we add `"filename_tmpl='{:05}.jpg'"` to the dict of `data.train`, `data.val` and `data.test` in the config files related with jester like this:

```
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
```

### Step 3. Extract Flow

This part is **optional** if you only want to use RGB frames.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/jester_extracted/
ln -s /mnt/SSD/jester_extracted/ ../../../data/jester/rawframes
```

Then, you can run the following script to extract optical flow based on RGB frames.

```shell
cd $MMACTION2/tools/data/jester/
bash extract_flow.sh
```

### Step 4. Encode Videos

This part is **optional** if you only want to use RGB frames.

You can run the following script to encode videos.

```shell
cd $MMACTION2/tools/data/jester/
bash encode_videos.sh
```

### Step 5. Generate File List

You can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION2/tools/data/jester/
bash generate_{rawframes, videos}_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for Jester preparation,
you will get the rawframes (RGB + Flow), and annotation files for Jester.

In the context of the whole project (for Jester only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── jester
│   │   ├── jester_{train,val}_list_rawframes.txt
│   │   ├── jester_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── 00001.jpg
│   |   |   |   ├── 00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```

For training and evaluating on Jester, please refer to [getting_started.md](/docs/getting_started.md).

## JHMDB

### Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{Jhuang:ICCV:2013,
    title = {Towards understanding action recognition},
    author = {H. Jhuang and J. Gall and S. Zuffi and C. Schmid and M. J. Black},
    booktitle = {International Conf. on Computer Vision (ICCV)},
    month = Dec,
    pages = {3192-3199},
    year = {2013}
}
```

For basic dataset information, you can refer to the dataset [website](http://jhmdb.is.tue.mpg.de/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/jhmdb/`.

### Download and Extract

You can download the RGB frames, optical flow and ground truth annotations from [google drive](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct).
The data are provided from [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md), which is adapted from [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector).

After downloading the `JHMDB.tar.gz` file and put it in `$MMACTION2/tools/data/jhmdb/`, you can run the following command to extract.

```shell
tar -zxvf JHMDB.tar.gz
```

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/JHMDB/
ln -s /mnt/SSD/JHMDB/ ../../../data/jhmdb
```

### Check Directory Structure

After extracting, you will get the `FlowBrox04` directory, `Frames` directory and `JHMDB-GT.pkl` for JHMDB.

In the context of the whole project (for JHMDB only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── jhmdb
│   |   ├── FlowBrox04
│   |   |   ├── brush_hair
│   |   |   |   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00039.jpg
│   |   |   |   |   ├── 00040.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── Trannydude___Brushing_SyntheticHair___OhNOES!__those_fukin_knots!_brush_hair_u_nm_np1_fr_goo_2
│   |   |   ├── ...
│   |   |   ├── wave
│   |   |   |   ├── 21_wave_u_nm_np1_fr_goo_5
│   |   |   |   ├── ...
│   |   |   |   ├── Wie_man_winkt!!_wave_u_cm_np1_fr_med_0
│   |   ├── Frames
│   |   |   ├── brush_hair
│   |   |   |   ├── April_09_brush_hair_u_nm_np1_ba_goo_0
│   |   |   |   |   ├── 00001.png
│   |   |   |   |   ├── 00002.png
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00039.png
│   |   |   |   |   ├── 00040.png
│   |   |   |   ├── ...
│   |   |   |   ├── Trannydude___Brushing_SyntheticHair___OhNOES!__those_fukin_knots!_brush_hair_u_nm_np1_fr_goo_2
│   |   |   ├── ...
│   |   |   ├── wave
│   |   |   |   ├── 21_wave_u_nm_np1_fr_goo_5
│   |   |   |   ├── ...
│   |   |   |   ├── Wie_man_winkt!!_wave_u_cm_np1_fr_med_0
│   |   ├── JHMDB-GT.pkl

```

:::{note}
The `JHMDB-GT.pkl` exists as a cache, it contains 6 items as follows:

1. `labels` (list): List of the 21 labels.
2. `gttubes` (dict): Dictionary that contains the ground truth tubes for each video.
   A **gttube** is dictionary that associates with each index of label and a list of tubes.
   A **tube** is a numpy array with `nframes` rows and 5 columns, each col is in format like `<frame index> <x1> <y1> <x2> <y2>`.
3. `nframes` (dict): Dictionary that contains the number of frames for each video, like `'walk/Panic_in_the_Streets_walk_u_cm_np1_ba_med_5': 16`.
4. `train_videos` (list): A list with `nsplits=1` elements, each one containing the list of training videos.
5. `test_videos` (list): A list with `nsplits=1` elements, each one containing the list of testing videos.
6. `resolution` (dict): Dictionary that outputs a tuple (h,w) of the resolution for each video, like `'pour/Bartender_School_Students_Practice_pour_u_cm_np1_fr_med_1': (240, 320)`.

:::

## Kinetics-\[400/600/700\]

### Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}
```

For basic dataset information, please refer to the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/). The scripts can be used for preparing kinetics400, kinetics600, kinetics700. To prepare different version of kinetics, you need to replace `${DATASET}` in the following examples with the specific dataset name. The choices of dataset names are `kinetics400`, `kinetics600` and `kinetics700`.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/${DATASET}/`.

:::{note}
Because of the expirations of some YouTube links, the sizes of kinetics dataset copies may be different. Here are the sizes of our kinetics dataset copies that used to train all checkpoints.

|   Dataset   | training videos | validation videos |
| :---------: | :-------------: | :---------------: |
| kinetics400 |     240436      |       19796       |

:::

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations by downloading from the official [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/).

```shell
bash download_annotations.sh ${DATASET}
```

Since some video urls are invalid, the number of video items in current official annotations are less than the original official ones.
So we provide an alternative way to download the older one as a reference.
Among these, the annotation files of Kinetics400 and Kinetics600 are from [official crawler](https://github.com/activitynet/ActivityNet/tree/199c9358907928a47cdfc81de4db788fddc2f91d/Crawler/Kinetics/data),
the annotation files of Kinetics700 are from [website](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) downloaded in 05/02/2021.

```shell
bash download_backup_annotations.sh ${DATASET}
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.

```shell
bash download_videos.sh ${DATASET}
```

**Important**: If you have already downloaded video dataset using the download script above,
you must replace all whitespaces in the class name for ease of processing by running

```shell
bash rename_classnames.sh ${DATASET}
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```bash
python ../resize_videos.py ../../../data/${DATASET}/videos_train/ ../../../data/${DATASET}/videos_train_256p_dense_cache --dense --level 2
```

You can also download from [Academic Torrents](https://academictorrents.com/) ([kinetics400](https://academictorrents.com/details/184d11318372f70018cf9a72ef867e2fb9ce1d26) & [kinetics700](https://academictorrents.com/details/49f203189fb69ae96fb40a6d0e129949e1dfec98) with short edge 256 pixels are available) and [cvdfoundation/kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) (Host by Common Visual Data Foundation and Kinetics400/Kinetics600/Kinetics-700-2020 are available)

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/${DATASET}_extracted_train/
ln -s /mnt/SSD/${DATASET}_extracted_train/ ../../../data/${DATASET}/rawframes_train/
mkdir /mnt/SSD/${DATASET}_extracted_val/
ln -s /mnt/SSD/${DATASET}_extracted_val/ ../../../data/${DATASET}/rawframes_val/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh ${DATASET}
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh ${DATASET}
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh ${DATASET}
```

The commands above can generate images with new short edge 256. If you want to generate images with short edge 320 (320p), or with fix size 340x256, you can change the args `--new-short 256` to `--new-short 320` or `--new-width 340 --new-height 256`.
More details can be found in [data_preparation](/docs/data_preparation.md)

### Step 4. Generate File List

you can run the follow scripts to generate file list in the format of videos and rawframes, respectively.

```shell
bash generate_videos_filelist.sh ${DATASET}
## execute the command below when rawframes are ready
bash generate_rawframes_filelist.sh ${DATASET}
```

### Step 5. Folder Structure

After the whole data pipeline for Kinetics preparation.
you can get the rawframes (RGB + Flow), videos and annotation files for Kinetics.

In the context of the whole project (for Kinetics only), the *minimal* folder structure will look like:
(*minimal* means that some data are not necessary: for example, you may want to evaluate kinetics using the original video format.)

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ${DATASET}
│   │   ├── ${DATASET}_train_list_videos.txt
│   │   ├── ${DATASET}_val_list_videos.txt
│   │   ├── annotations
│   │   ├── videos_train
│   │   ├── videos_val
│   │   │   ├── abseiling
│   │   │   │   ├── 0wR5jVB-WPk_000417_000427.mp4
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── wrapping_present
│   │   │   ├── ...
│   │   │   ├── zumba
│   │   ├── rawframes_train
│   │   ├── rawframes_val

```

For training and evaluating on Kinetics, please refer to [getting_started](/docs/getting_started.md).

## Moments in Time

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{monfortmoments,
    title={Moments in Time Dataset: one million videos for event understanding},
    author={Monfort, Mathew and Andonian, Alex and Zhou, Bolei and Ramakrishnan, Kandan and Bargal, Sarah Adel and Yan, Tom and Brown, Lisa and Fan, Quanfu and Gutfruend, Dan and Vondrick, Carl and others},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    year={2019},
    issn={0162-8828},
    pages={1--8},
    numpages={8},
    doi={10.1109/TPAMI.2019.2901464},
}
```

For basic dataset information, you can refer to the dataset [website](http://moments.csail.mit.edu/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/mit/`.

### Step 1. Prepare Annotations and Videos

First of all, you have to visit the official [website](http://moments.csail.mit.edu/), fill in an application form for downloading the dataset. Then you will get the download link. You can use `bash preprocess_data.sh` to prepare annotations and videos. However, the download command is missing in that script. Remember to download the dataset to the proper place follow the comment in this script.

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```shell
python ../resize_videos.py ../../../data/mit/videos/ ../../../data/mit/videos_256p_dense_cache --dense --level 2
```

### Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. And you can run the following script to soft link the extracted frames.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mit_extracted/
ln -s /mnt/SSD/mit_extracted/ ../../../data/mit/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
bash extract_frames.sh
```

### Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_{rawframes, videos}_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Moments in Time.

In the context of the whole project (for Moments in Time only), the folder structure will look like:

```
mmaction2
├── data
│   └── mit
│       ├── annotations
│       │   ├── license.txt
│       │   ├── moments_categories.txt
│       │   ├── README.txt
│       │   ├── trainingSet.csv
│       │   └── validationSet.csv
│       ├── mit_train_rawframe_anno.txt
│       ├── mit_train_video_anno.txt
│       ├── mit_val_rawframe_anno.txt
│       ├── mit_val_video_anno.txt
│       ├── rawframes
│       │   ├── training
│       │   │   ├── adult+female+singing
│       │   │   │   ├── 0P3XG_vf91c_35
│       │   │   │   │   ├── flow_x_00001.jpg
│       │   │   │   │   ├── flow_x_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── flow_y_00001.jpg
│       │   │   │   │   ├── flow_y_00002.jpg
│       │   │   │   │   ├── ...
│       │   │   │   │   ├── img_00001.jpg
│       │   │   │   │   └── img_00002.jpg
│       │   │   │   └── yt-zxQfALnTdfc_56
│       │   │   │   │   ├── ...
│       │   │   └── yawning
│       │   │       ├── _8zmP1e-EjU_2
│       │   │       │   ├── ...
│       │   └── validation
│       │   │       ├── ...
│       └── videos
│           ├── training
│           │   ├── adult+female+singing
│           │   │   ├── 0P3XG_vf91c_35.mp4
│           │   │   ├── ...
│           │   │   └── yt-zxQfALnTdfc_56.mp4
│           │   └── yawning
│           │       ├── ...
│           └── validation
│           │   ├── ...
└── mmaction
└── ...

```

For training and evaluating on Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).

## Multi-Moments in Time

### Introduction

<!-- [DATASET] -->

```BibTeX
@misc{monfort2019multimoments,
    title={Multi-Moments in Time: Learning and Interpreting Models for Multi-Action Video Understanding},
    author={Mathew Monfort and Kandan Ramakrishnan and Alex Andonian and Barry A McNamara and Alex Lascelles, Bowen Pan, Quanfu Fan, Dan Gutfreund, Rogerio Feris, Aude Oliva},
    year={2019},
    eprint={1911.00232},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [website](http://moments.csail.mit.edu).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/mmit/`.

### Step 1. Prepare Annotations and Videos

First of all, you have to visit the official [website](http://moments.csail.mit.edu/), fill in an application form for downloading the dataset. Then you will get the download link. You can use `bash preprocess_data.sh` to prepare annotations and videos. However, the download command is missing in that script. Remember to download the dataset to the proper place follow the comment in this script.

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_videos.py ../../../data/mmit/videos/ ../../../data/mmit/videos_256p_dense_cache --dense --level 2
```

### Step 2. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

First, you can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/mmit_extracted/
ln -s /mnt/SSD/mmit_extracted/ ../../../data/mmit/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

### Step 3. Generate File List

you can run the follow script to generate file list in the format of rawframes or videos.

```shell
bash generate_rawframes_filelist.sh
bash generate_videos_filelist.sh
```

### Step 4. Check Directory Structure

After the whole data process for Multi-Moments in Time preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Multi-Moments in Time.

In the context of the whole project (for Multi-Moments in Time only), the folder structure will look like:

```
mmaction2/
└── data
    └── mmit
        ├── annotations
        │   ├── moments_categories.txt
        │   ├── trainingSet.txt
        │   └── validationSet.txt
        ├── mmit_train_rawframes.txt
        ├── mmit_train_videos.txt
        ├── mmit_val_rawframes.txt
        ├── mmit_val_videos.txt
        ├── rawframes
        │   ├── 0-3-6-2-9-1-2-6-14603629126_5
        │   │   ├── flow_x_00001.jpg
        │   │   ├── flow_x_00002.jpg
        │   │   ├── ...
        │   │   ├── flow_y_00001.jpg
        │   │   ├── flow_y_00002.jpg
        │   │   ├── ...
        │   │   ├── img_00001.jpg
        │   │   └── img_00002.jpg
        │   │   ├── ...
        │   └── yt-zxQfALnTdfc_56
        │   │   ├── ...
        │   └── ...

        └── videos
            └── adult+female+singing
                ├── 0-3-6-2-9-1-2-6-14603629126_5.mp4
                └── yt-zxQfALnTdfc_56.mp4
            └── ...
```

For training and evaluating on Multi-Moments in Time, please refer to [getting_started.md](/docs/getting_started.md).

## OmniSource

### Introduction

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

|  Dataset Name   | ##samples |  Size   |  Teacher Model   | #samples after filtering | #samples similar to k200_val |
| :-------------: | :-------: | :-----: | :--------------: | :----------------------: | :--------------------------: |
|   k200_train    |   76030   |  45.6G  |       N/A        |           N/A            |             N/A              |
|    k200_val     |   4838    |  2.9G   |       N/A        |           N/A            |             N/A              |
| googleimage_200 |  3050880  | 265.5G  |   TSN-R50-8seg   |         1188695          |             967              |
|  insimage_200   |  3654650  | 224.4G  |   TSN-R50-8seg   |          879726          |             116              |
|  insvideo_200   |  732855   | 1487.6G | SlowOnly-8x8-R50 |          330680          |             956              |
| k200_raw_train  |   76027   | 963.5G  | SlowOnly-8x8-R50 |           N/A            |             N/A              |

The file structure of our uploaded OmniSource dataset looks like:

```
OmniSource/
├── annotations
│   ├── googleimage_200
│   │   ├── googleimage_200.txt                       File list of all valid images crawled from Google.
│   │   ├── tsn_8seg_googleimage_200_duplicate.txt    Positive file list of images crawled from Google, which is similar to a validation example.
│   │   ├── tsn_8seg_googleimage_200.txt              Positive file list of images crawled from Google, filtered by the teacher model.
│   │   └── tsn_8seg_googleimage_200_wodup.txt        Positive file list of images crawled from Google, filtered by the teacher model, after de-duplication.
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

### Data Preparation

For data preparation, you need to first download those data. For `kinetics_200` and 3 web datasets: `googleimage_200`, `insimage_200` and `insvideo_200`, you just need to extract each volume and merge their contents.

For Kinetics raw videos, since loading long videos is very heavy, you need to first trim it into clips. Here we provide a script named `trim_raw_video.py`. It trims a long video into 10-second clips and remove the original raw video. You can use it to trim the Kinetics raw video.

The data should be placed in `data/OmniSource/`. When data preparation finished, the folder structure of `data/OmniSource` looks like (We omit the files not needed in training & testing for simplicity):

```
data/OmniSource/
├── annotations
│   ├── googleimage_200
│   │   └── tsn_8seg_googleimage_200_wodup.txt    Positive file list of images crawled from Google, filtered by the teacher model, after de-duplication.
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

## Skeleton Dataset

<!-- [DATASET] -->

```BibTeX
@misc{duan2021revisiting,
      title={Revisiting Skeleton-based Action Recognition},
      author={Haodong Duan and Yue Zhao and Kai Chen and Dian Shao and Dahua Lin and Bo Dai},
      year={2021},
      eprint={2104.13586},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Introduction

We release the skeleton annotations used in [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586). By default, we use [Faster-RCNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) with ResNet50 backbone for human detection and [HRNet-w32](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py) for single person pose estimation. For FineGYM, we use Ground-Truth bounding boxes for the athlete instead of detection bounding boxes.

### Prepare Annotations

We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.

- NTURGB+D \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
- NTURGB+D \[3D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_3d.pkl
- NTURGB+D 120 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl
- NTURGB+D 120 \[3D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_3d.pkl
- GYM \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/gym_2d.pkl
  - GYM 2D skeletons are extracted with ground-truth human bounding boxes, which can be downloaded with [link](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_gt_bboxes.pkl). Please cite [PoseConv3D](https://arxiv.org/abs/2104.13586) if you use it in your project.
- UCF101 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
- HMDB51 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
- Diving48 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/diving48_2d.pkl
- Kinetics400 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/k400_2d.pkl (Table of contents only, no skeleton annotations)

For Kinetics400, since the skeleton annotations are large, we do not provide the direct download links on aliyun. Please use the following link to download the `kpfiles` and extract it under `$MMACTION2/data/k400` for Kinetics400 training & testing: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EeyDCVskqLtClMVVwqD53acBF2FEwkctp3vtRbkLfnKSTw?e=B3SZlM

If you want to generate 2D skeleton annotations of specified video, please install mmdetection and mmpose first, then use the following script to extract skeleton annotations of NTURGB+D video:

```python
python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001.pkl
```

please note that, due to the upgrade of mmpose, the inference results may have slight differences from the provided skeleton annotations.

### The Format of Annotations

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple\[int\]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple\[int\]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape \[M x T x V x C\]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape \[M x T x V\]): The confidence score of keypoints. Only required for 2D skeletons.

### Visualization

For skeleton data visualization, you need also to prepare the RGB videos. Please refer to [visualize_heatmap_volume](/demo/visualize_heatmap_volume.ipynb) for detailed process. Here we provide some visualization examples from NTU-60 and FineGYM.

<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> Pose Estimation Results </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529341-6fc95080-a90f-11eb-8f0d-57fdb35d1ba4.gif" width="455"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531676-04cd4900-a912-11eb-8db4-a93343bedd01.gif" width="455"/>
</div></td>
    <td>
<div align="center">
  <b> Keypoint Heatmap Volume Visualization </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529336-6dff8d00-a90f-11eb-807e-4d9168997655.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531658-00a12b80-a912-11eb-957b-561c280a86da.gif" width="256"/>
</div></td>
    <td>
<div align="center">
  <b> Limb Heatmap Volume Visualization </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529322-6a6c0600-a90f-11eb-81df-6fbb36230bd0.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531649-fed76800-a911-11eb-8ca9-0b4e58f43ad9.gif" width="256"/>
</div></td>
  </tr>
</thead>
</table>

### Convert the NTU RGB+D raw skeleton data to our format (only applicable to GCN backbones)

Here we also provide the script for converting the NTU RGB+D raw skeleton data to our format.
First, download the raw skeleton data of NTU-RGBD 60 and NTU-RGBD 120 from https://github.com/shahroudy/NTURGB-D.

For NTU-RGBD 60, preprocess data and convert the data format with

```python
python gen_ntu_rgbd_raw.py --data-path your_raw_nturgbd60_skeleton_path --ignored-sample-path NTU_RGBD_samples_with_missing_skeletons.txt --out-folder your_nturgbd60_output_path --task ntu60
```

For NTU-RGBD 120, preprocess data and convert the data format with

```python
python gen_ntu_rgbd_raw.py --data-path your_raw_nturgbd120_skeleton_path --ignored-sample-path NTU_RGBD120_samples_with_missing_skeletons.txt --out-folder your_nturgbd120_output_path --task ntu120
```

### Convert annotations from third-party projects

We provide scripts to convert skeleton annotations from third-party projects to MMAction2 formats:

- BABEL: `babel2mma2.py`

**TODO**:

- [x] FineGYM
- [x] NTU60_XSub
- [x] NTU120_XSub
- [x] NTU60_XView
- [x] NTU120_XSet
- [x] UCF101
- [x] HMDB51
- [x] Kinetics

## Something-Something V1

### Introduction

<!-- [DATASET] -->

```BibTeX
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [paper](https://arxiv.org/pdf/1706.04261.pdf).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/sthv1/`.

### Step 1. Prepare Annotations

Since the official [website](https://20bn.com/datasets/something-something/v1) of Something-Something V1 is currently unavailable, you can download the annotations from third-part source to `$MMACTION2/data/sthv1/` .

### Step 2. Prepare RGB Frames

Since the official dataset doesn't provide the original video data and only extracted RGB frames are available, you have to directly download RGB frames.

You can download all compressed file parts from third-part source  to `$MMACTION2/data/sthv1/` and use the following command to uncompress.

```shell
cd $MMACTION2/data/sthv1/
cat 20bn-something-something-v1-?? | tar zx
cd $MMACTION2/tools/data/sthv1/
```

For users who only want to use RGB frames, you can skip to step 5 to generate file lists in the format of rawframes.
Since the prefix of official JPGs is "%05d.jpg" (e.g., "00001.jpg"), users need to add `"filename_tmpl='{:05}.jpg'"` to the dict of `data.train`, `data.val` and `data.test` in the config files related with sthv1 like this:

```
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
```

### Step 3. Extract Flow

This part is **optional** if you only want to use RGB frames.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv1_extracted/
ln -s /mnt/SSD/sthv1_extracted/ ../../../data/sthv1/rawframes
```

Then, you can run the following script to extract optical flow based on RGB frames.

```shell
cd $MMACTION2/tools/data/sthv1/
bash extract_flow.sh
```

### Step 4. Encode Videos

This part is **optional** if you only want to use RGB frames.

You can run the following script to encode videos.

```shell
cd $MMACTION2/tools/data/sthv1/
bash encode_videos.sh
```

### Step 5. Generate File List

You can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION2/tools/data/sthv1/
bash generate_{rawframes, videos}_filelist.sh
```

### Step 6. Check Directory Structure

After the whole data process for Something-Something V1 preparation,
you will get the rawframes (RGB + Flow), and annotation files for Something-Something V1.

In the context of the whole project (for Something-Something V1 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv1
│   │   ├── sthv1_{train,val}_list_rawframes.txt
│   │   ├── sthv1_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── 00001.jpg
│   |   |   |   ├── 00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```

For training and evaluating on Something-Something V1, please refer to [getting_started.md](/docs/getting_started.md).

## Something-Something V2

### Introduction

<!-- [DATASET] -->

```BibTeX
@misc{goyal2017something,
      title={The "something something" video database for learning and evaluating visual common sense},
      author={Raghav Goyal and Samira Ebrahimi Kahou and Vincent Michalski and Joanna Materzyńska and Susanne Westphal and Heuna Kim and Valentin Haenel and Ingo Fruend and Peter Yianilos and Moritz Mueller-Freitag and Florian Hoppe and Christian Thurau and Ingo Bax and Roland Memisevic},
      year={2017},
      eprint={1706.04261},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For basic dataset information, you can refer to the dataset [website](https://developer.qualcomm.com/software/ai-datasets/something-something).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/sthv2/`.

### Step 1. Prepare Annotations

First of all, you have to sign in and download annotations to `$MMACTION2/data/sthv2/annotations` on the official [website](https://20bn.com/datasets/something-something/v2).

### Step 2. Prepare Videos

Then, you can download all data parts to `$MMACTION2/data/sthv2/` and use the following command to uncompress.

```shell
cd $MMACTION2/data/sthv2/
cat 20bn-something-something-v2-?? | tar zx
cd $MMACTION2/tools/data/sthv2/
```

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/sthv2_extracted/
ln -s /mnt/SSD/sthv2_extracted/ ../../../data/sthv2/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/sthv2/
bash extract_frames.sh
```

### Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
cd $MMACTION2/tools/data/sthv2/
bash generate_{rawframes, videos}_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for Something-Something V2 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for Something-Something V2.

In the context of the whole project (for Something-Something V2 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── sthv2
│   │   ├── sthv2_{train,val}_list_rawframes.txt
│   │   ├── sthv2_{train,val}_list_videos.txt
│   │   ├── annotations
│   |   ├── videos
│   |   |   ├── 1.mp4
│   |   |   ├── 2.mp4
│   |   |   ├──...
│   |   ├── rawframes
│   |   |   ├── 1
│   |   |   |   ├── img_00001.jpg
│   |   |   |   ├── img_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_x_00001.jpg
│   |   |   |   ├── flow_x_00002.jpg
│   |   |   |   ├── ...
│   |   |   |   ├── flow_y_00001.jpg
│   |   |   |   ├── flow_y_00002.jpg
│   |   |   |   ├── ...
│   |   |   ├── 2
│   |   |   ├── ...

```

For training and evaluating on Something-Something V2, please refer to [getting_started.md](/docs/getting_started.md).
s/getting_started.md).

## THUMOS'14

### Introduction

<!-- [DATASET] -->

```BibTeX
@misc{THUMOS14,
    author = {Jiang, Y.-G. and Liu, J. and Roshan Zamir, A. and Toderici, G. and Laptev,
    I. and Shah, M. and Sukthankar, R.},
    title = {{THUMOS} Challenge: Action Recognition with a Large
    Number of Classes},
    howpublished = "\url{http://crcv.ucf.edu/THUMOS14/}",
    Year = {2014}
}
```

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/THUMOS14/download.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/thumos14/`.

### Step 1. Prepare Annotations

First of all, run the following script to prepare annotations.

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
cd $MMACTION2/tools/data/thumos14/
bash download_videos.sh
```

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/thumos14_extracted/
ln -s /mnt/SSD/thumos14_extracted/ ../data/thumos14/rawframes/
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_rgb_frames_opencv.sh
```

If both are required, run the following script to extract frames.

```shell
cd $MMACTION2/tools/data/thumos14/
bash extract_frames.sh tvl1
```

### Step 4. Fetch File List

This part is **optional** if you do not use SSN model.

You can run the follow script to fetch pre-computed tag proposals.

```shell
cd $MMACTION2/tools/data/thumos14/
bash fetch_tag_proposals.sh
```

### Step 5. Denormalize Proposal File

This part is **optional** if you do not use SSN model.

You can run the follow script to denormalize pre-computed tag proposals according to
actual number of local rawframes.

```shell
cd $MMACTION2/tools/data/thumos14/
bash denormalize_proposal_file.sh
```

### Step 6. Check Directory Structure

After the whole data process for THUMOS'14 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for THUMOS'14.

In the context of the whole project (for THUMOS'14 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── thumos14
│   │   ├── proposals
│   │   |   ├── thumos14_tag_val_normalized_proposal_list.txt
│   │   |   ├── thumos14_tag_test_normalized_proposal_list.txt
│   │   ├── annotations_val
│   │   ├── annotations_test
│   │   ├── videos
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001.mp4
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001.mp4
│   │   │   |   ├── ...
│   │   ├── rawframes
│   │   │   ├── val
│   │   │   |   ├── video_validation_0000001
|   │   │   |   │   ├── img_00001.jpg
|   │   │   |   │   ├── img_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_x_00001.jpg
|   │   │   |   │   ├── flow_x_00002.jpg
|   │   │   |   │   ├── ...
|   │   │   |   │   ├── flow_y_00001.jpg
|   │   │   |   │   ├── flow_y_00002.jpg
|   │   │   |   │   ├── ...
│   │   │   |   ├── ...
│   │   |   ├── test
│   │   │   |   ├── video_test_0000001
```

For training and evaluating on THUMOS'14, please refer to [getting_started.md](/docs/getting_started.md).

## UCF101-24

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

For basic dataset information, you can refer to the dataset [website](http://www.thumos.info/download.html).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ucf101_24/`.

### Download and Extract

You can download the RGB frames, optical flow and ground truth annotations from [google drive](https://drive.google.com/drive/folders/1BvGywlAGrACEqRyfYbz3wzlVV3cDFkct).
The data are provided from [MOC](https://github.com/MCG-NJU/MOC-Detector/blob/master/readme/Dataset.md), which is adapted from [act-detector](https://github.com/vkalogeiton/caffe/tree/act-detector) and [corrected-UCF101-Annots](https://github.com/gurkirt/corrected-UCF101-Annots).

:::{note}
The annotation of this UCF101-24 is from [here](https://github.com/gurkirt/corrected-UCF101-Annots), which is more correct.
:::

After downloading the `UCF101_v2.tar.gz` file and put it in `$MMACTION2/tools/data/ucf101_24/`, you can run the following command to uncompress.

```shell
tar -zxvf UCF101_v2.tar.gz
```

### Check Directory Structure

After uncompressing, you will get the `rgb-images` directory, `brox-images` directory and `UCF101v2-GT.pkl` for UCF101-24.

In the context of the whole project (for UCF101-24 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101_24
│   |   ├── brox-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── rgb-images
│   |   |   ├── Basketball
│   |   |   |   ├── v_Basketball_g01_c01
│   |   |   |   |   ├── 00001.jpg
│   |   |   |   |   ├── 00002.jpg
│   |   |   |   |   ├── ...
│   |   |   |   |   ├── 00140.jpg
│   |   |   |   |   ├── 00141.jpg
│   |   |   ├── ...
│   |   |   ├── WalkingWithDog
│   |   |   |   ├── v_WalkingWithDog_g01_c01
│   |   |   |   ├── ...
│   |   |   |   ├── v_WalkingWithDog_g25_c04
│   |   ├── UCF101v2-GT.pkl

```

:::{note}
The `UCF101v2-GT.pkl` exists as a cache, it contains 6 items as follows:
:::

1. `labels` (list): List of the 24 labels.
2. `gttubes` (dict): Dictionary that contains the ground truth tubes for each video.
   A **gttube** is dictionary that associates with each index of label and a list of tubes.
   A **tube** is a numpy array with `nframes` rows and 5 columns, each col is in format like `<frame index> <x1> <y1> <x2> <y2>`.
3. `nframes` (dict): Dictionary that contains the number of frames for each video, like `'HorseRiding/v_HorseRiding_g05_c02': 151`.
4. `train_videos` (list): A list with `nsplits=1` elements, each one containing the list of training videos.
5. `test_videos` (list): A list with `nsplits=1` elements, each one containing the list of testing videos.
6. `resolution` (dict): Dictionary that outputs a tuple (h,w) of the resolution for each video, like `'FloorGymnastics/v_FloorGymnastics_g09_c03': (240, 320)`.

## UCF-101

### Introduction

<!-- [DATASET] -->

```BibTeX
@article{Soomro2012UCF101AD,
  title={UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild},
  author={K. Soomro and A. Zamir and M. Shah},
  journal={ArXiv},
  year={2012},
  volume={abs/1212.0402}
}
```

For basic dataset information, you can refer to the dataset [website](https://www.crcv.ucf.edu/research/data-sets/ucf101/).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ucf101/`.

### Step 1. Prepare Annotations

First of all, you can run the following script to prepare annotations.

```shell
bash download_annotations.sh
```

### Step 2. Prepare Videos

Then, you can run the following script to prepare videos.

```shell
bash download_videos.sh
```

For better decoding speed, you can resize the original videos into smaller sized, densely encoded version by:

```
python ../resize_videos.py ../../../data/ucf101/videos/ ../../../data/ucf101/videos_256p_dense_cache --dense --level 2 --ext avi
```

### Step 3. Extract RGB and Flow

This part is **optional** if you only want to use the video loader.

Before extracting, please refer to [install.md](/docs/install.md) for installing [denseflow](https://github.com/open-mmlab/denseflow).

If you have plenty of SSD space, then we recommend extracting frames there for better I/O performance. The extracted frames (RGB + Flow) will take up about 100GB.

You can run the following script to soft link SSD.

```shell
## execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/ucf101_extracted/
ln -s /mnt/SSD/ucf101_extracted/ ../../../data/ucf101/rawframes
```

If you only want to play with RGB frames (since extracting optical flow can be time-consuming), consider running the following script to extract **RGB-only** frames using denseflow.

```shell
bash extract_rgb_frames.sh
```

If you didn't install denseflow, you can still extract RGB frames using OpenCV by the following script, but it will keep the original size of the images.

```shell
bash extract_rgb_frames_opencv.sh
```

If Optical Flow is also required, run the following script to extract flow using "tvl1" algorithm.

```shell
bash extract_frames.sh
```

### Step 4. Generate File List

you can run the follow script to generate file list in the format of rawframes and videos.

```shell
bash generate_videos_filelist.sh
bash generate_rawframes_filelist.sh
```

### Step 5. Check Directory Structure

After the whole data process for UCF-101 preparation,
you will get the rawframes (RGB + Flow), videos and annotation files for UCF-101.

In the context of the whole project (for UCF-101 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf101
│   │   ├── ucf101_{train,val}_split_{1,2,3}_rawframes.txt
│   │   ├── ucf101_{train,val}_split_{1,2,3}_videos.txt
│   │   ├── annotations
│   │   ├── videos
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01.avi

│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g25_c05.avi
│   │   ├── rawframes
│   │   │   ├── ApplyEyeMakeup
│   │   │   │   ├── v_ApplyEyeMakeup_g01_c01
│   │   │   │   │   ├── img_00001.jpg
│   │   │   │   │   ├── img_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_x_00001.jpg
│   │   │   │   │   ├── flow_x_00002.jpg
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── flow_y_00001.jpg
│   │   │   │   │   ├── flow_y_00002.jpg
│   │   │   ├── ...
│   │   │   ├── YoYo
│   │   │   │   ├── v_YoYo_g01_c01
│   │   │   │   ├── ...
│   │   │   │   ├── v_YoYo_g25_c05

```

For training and evaluating on UCF-101, please refer to [getting_started.md](/docs/getting_started.md).
