# Preparing Skeleton Dataset

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

## Introduction

We release the skeleton annotations used in [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586). By default, we use [Faster-RCNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) with ResNet50 backbone for human detection and [HRNet-w32](https://github.com/open-mmlab/mmpose/blob/master/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py) for single person pose estimation. For FineGYM, we use Ground-Truth bounding boxes for the athlete instead of detection bounding boxes. Currently, we release the skeleton annotations for FineGYM and NTURGB-D Xsub split. Other annotations will be soo released.

## Prepare Annotations

Currently, we support HMDB51, UCF101, FineGYM and NTURGB+D. For FineGYM, you can execute following scripts to prepare the annotations.

```shell
bash download_annotations.sh ${DATASET}
```

Due to [Conditions of Use](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) of the NTURGB+D dataset, we can not directly release the annotations used in our experiments. So that we provide a script to generate pose annotations for videos in NTURGB+D datasets, which generate a dictionary and save it as a single pickle file. You can create a list which contain all annotation dictionaries of corresponding videos and save them as a pickle file. Then you can get the `ntu60_xsub_train.pkl`, `ntu60_xsub_val.pkl`, `ntu120_xsub_train.pkl`, `ntu120_xsub_val.pkl` that we used in training.

For those who have not enough computations for pose extraction, we provide the outputs of the above pipeline here, corresponding to 4 different splits of NTURGB+D datasets:

- ntu60_xsub_train: https://download.openmmlab.com/mmaction/posec3d/ntu60_xsub_train.pkl
- ntu60_xsub_val: https://download.openmmlab.com/mmaction/posec3d/ntu60_xsub_val.pkl
- ntu120_xsub_train: https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_train.pkl
- ntu120_xsub_val: https://download.openmmlab.com/mmaction/posec3d/ntu120_xsub_val.pkl
- hmdb51: https://download.openmmlab.com/mmaction/posec3d/hmdb51.pkl
- ucf101: https://download.openmmlab.com/mmaction/posec3d/ucf101.pkl

To generate 2D pose annotations for a single video, first, you need to install mmdetection and mmpose from src code. After that, you need to replace the placeholder `mmdet_root` and `mmpose_root` in `ntu_pose_extraction.py` with your installation path. Then you can use following scripts for NTURGB+D video pose extraction:

```python
python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001.pkl
```

After you get pose annotations for all videos in a dataset split, like `ntu60_xsub_val`. You can gather them into a single list and save the list as `ntu60_xsub_val.pkl`. You can use those larger pickle files for training and testing.

## The Format of PoseC3D Annotations

Here we briefly introduce the format of PoseC3D Annotations, we will take `gym_train.pkl` as an example: the content of `gym_train.pkl` is a list of length 20484, each item is a dictionary that is the skeleton annotation of one video. Each dictionary has following fields:

- keypoint: The keypoint coordinates, which is a numpy array of the shape N (#person) x T (temporal length) x K (#keypoints, 17 in our case) x 2 (x, y coordinate).
- keypoint_score:  The keypoint confidence scores, which is a numpy array of the shape N (#person) x T (temporal length) x K (#keypoints, 17 in our case).
- frame_dir: The corresponding video name.
- label: The action category.
- img_shape: The image shape of each frame.
- original_shape: Same as above.
- total_frames: The temporal length of the video.

For training with your custom dataset, you can refer to [Custom Dataset Training](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md).

## Visualization

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

## Convert the NTU RGB+D raw skeleton data to our format (only applicable to GCN backbones)

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

## Convert annotations from third-party projects

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
- [ ] Kinetics
