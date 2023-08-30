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

We release the skeleton annotations used in [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586). By default, we use [Faster-RCNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) with ResNet50 backbone for human detection and [HRNet-w32](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py) for single person pose estimation. For FineGYM, we use Ground-Truth bounding boxes for the athlete instead of detection bounding boxes.

## Prepare Annotations

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

For Kinetics400, since the skeleton annotations are large, we do not provide the direct download links on aliyun. Please use the following link to download the `k400_kpfiles_2d.zip` and extract it under `$MMACTION2/data/skeleton/kpfiles` for Kinetics400 training & testing: https://openxlab.org.cn/datasets/OpenMMLab/Kinetics400-skeleton

If you want to generate 2D skeleton annotations of specified video, please install mmdetection and mmpose first, then use the following script to extract skeleton annotations of NTURGB+D video:

```python
python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001.pkl
```

please note that, due to the upgrade of mmpose, the inference results may have slight differences from the provided skeleton annotations.

## The Format of Annotations

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

## Visualization

For skeleton data visualization, you need also to prepare the RGB videos. Please refer to \[visualize_heatmap_volume\] for detailed process. Here we provide some visualization examples from NTU-60 and FineGYM.

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
- [x] Kinetics
