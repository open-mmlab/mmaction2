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

Currently, we support one dataset: FineGYM. You can execute following scripts to prepare the annotations.

```shell
bash download_annotations.sh ${DATASET}
```

PS: Due to [Conditions of Use](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) of the NTURGB-D dataset, we can not directly release the annotations used in our experiments. We will prepare a script for pose annotation generation ASAP. Once accomplished, you can use this script to generate all pose annotations used in our experiments.

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

**TODO**:

- [x] FineGYM
- [ ] NTU60_XSub
- [ ] NTU120_XSub
- [ ] NTU60_XView
- [ ] NTU120_XSet
- [ ] Kinetics
