# 准备骨架数据集

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

## 简介

MMAction2 发布 [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586) 论文中所使用的骨架标注。
默认使用 [Faster-RCNN](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py) 作为人体检测器，
使用 [HRNet-w32](https://github.com/open-mmlab/mmpose/blob/master/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py) 作为单人姿态估计模型。
对于 FineGYM 数据集，MMAction2 使用的是运动员的真实框标注，而非检测器所出的框。目前，MMAction2 已发布 FineGYM 和 NTURGB-D Xsub 部分的骨架标注，其他数据集的标注也将很快发布。

## 准备标注文件

目前，MMAction2 支持 HMDB51, UCF101, FineGYM 和 NTURGB+D 数据集。对于 FineGYM 数据集，用户可以使用以下脚本下载标注文件。

```shell
bash download_annotations.sh ${DATASET}
```

由于 NTURGB+D 数据集的 [使用条例](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp)，MMAction2 并未直接发布实验中所使用的标注文件。
因此，这里提供生成 NTURGB+D 数据集中视频的姿态标注文件，这将生成一个 dict 数据并将其保存为一个 pickle 文件。
用户可以生成一个 list 用以包含对应视频的 dict 数据，并将其保存为一个 pickle 文件。
之后，用户可以获得 `ntu60_xsub_train.pkl`, `ntu60_xsub_val.pkl`, `ntu120_xsub_train.pkl`, `ntu120_xsub_val.pkl` 文件用于训练。

对于无法进行姿态提取的用户，这里提供了上述流程的输出结果，分别对应 NTURGB-D 数据集的 4 个部分：

- NTURGB+D \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_2d.pkl
- NTURGB+D \[3D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu60_3d.pkl
- NTURGB+D 120 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl
- NTURGB+D 120 \[3D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_3d.pkl
- GYM \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/gym_2d.pkl
  - GYM 2D 姿态标注文件是基于运动员的真实标注框生成的，用户可以从这个[链接](https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_gt_bboxes.pkl)下载真实标注框。如果你在项目中使用了该数据，请引用 [PoseConv3D](https://arxiv.org/abs/2104.13586)
- UCF101 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
- HMDB51 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/hmdb51_2d.pkl
- Diving48 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/diving48_2d.pkl
- Kinetics400 \[2D Skeleton\]: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/k400_2d.pkl (只包含数据列表，没有姿态标注文件)

由于 Kinetics400 数据集姿态标注文件过大，我们不提供阿里云的下载链接，请使用此[链接](https://openxlab.org.cn/datasets/OpenMMLab/Kinetics400-skeleton)下载 `k400_kpfiles_2d.zip`，解压到 `$MMACTION2/data/skeleton/kpfiles` 目录下，用于 Kinetics400 的训练和测试。

若想生成单个视频的 2D 姿态标注文件，用户在安装 mmdetection 和 mmpose 之后，可使用以下脚本进行 NTURGB+D 视频的姿态提取：

```python
python ntu_pose_extraction.py S001C001P001R001A001_rgb.avi S001C001P001R001A001.pkl
```

请注意，由于 mmpose 算法库升级，此脚本的推理结果与提供的姿态点数据集可能略有差异。

在用户获得数据集某部分所有视频的姿态标注文件（如 `ntu60_xsub_val`）后，可以将其集合成一个 list 数据并保存为 `ntu60_xsub_val.pkl`。用户可用这些大型 pickle 文件进行训练和测试。

## PoseC3D 的标注文件格式

这里简单介绍 PoseC3D 的标注文件格式。以 `gym_train.pkl` 为例：`gym_train.pkl` 存储一个长度为 20484 的 list，list 的每一项为单个视频的骨架标注 dict。每个 dict 的内容如下：

- keypoint：关键点坐标，大小为 N（#人数）x T（时序长度）x K（#关键点, 这里为17）x 2 （x，y 坐标）的 numpy array 数据类型
- keypoint_score：关键点的置信分数，大小为 N（#人数）x T（时序长度）x K（#关键点, 这里为17）的 numpy array 数据类型
- frame_dir: 对应视频名
- label: 动作类别
- img_shape: 每一帧图像的大小
- original_shape: 同 `img_shape`
- total_frames: 视频时序长度

如用户想使用自己的数据集训练 PoseC3D，可以参考 [Custom Dataset Training](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md)。

## 可视化

为了可视化骨架数据，用户需要准备 RGB 的视频。详情可参考 \[visualize_heatmap_volume\]。这里提供一些 NTU-60 和 FineGYM 上的例子

<table>
<thead>
  <tr>
    <td>
<div align="center">
  <b> 姿态估计结果 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529341-6fc95080-a90f-11eb-8f0d-57fdb35d1ba4.gif" width="455"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531676-04cd4900-a912-11eb-8db4-a93343bedd01.gif" width="455"/>
</div></td>
    <td>
<div align="center">
  <b> 关键点热力图三维可视化 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529336-6dff8d00-a90f-11eb-807e-4d9168997655.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531658-00a12b80-a912-11eb-957b-561c280a86da.gif" width="256"/>
</div></td>
    <td>
<div align="center">
  <b> 肢体热力图三维可视化 </b>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116529322-6a6c0600-a90f-11eb-81df-6fbb36230bd0.gif" width="256"/>
  <br/>
  <br/>
  <img src="https://user-images.githubusercontent.com/34324155/116531649-fed76800-a911-11eb-8ca9-0b4e58f43ad9.gif" width="256"/>
</div></td>
  </tr>
</thead>
</table>

## 如何将 NTU RGB+D 原始数据转化为 MMAction2 格式 （转换好的标注文件目前仅适用于 GCN 模型）

这里介绍如何将 NTU RGB+D 原始数据转化为 MMAction2 格式。首先，需要从 https://github.com/shahroudy/NTURGB-D 下载原始 NTU-RGBD 60 和 NTU-RGBD 120 数据集的原始骨架数据。

对于 NTU-RGBD 60 数据集，可使用以下脚本

```python
python gen_ntu_rgbd_raw.py --data-path your_raw_nturgbd60_skeleton_path --ignored-sample-path NTU_RGBD_samples_with_missing_skeletons.txt --out-folder your_nturgbd60_output_path --task ntu60
```

对于 NTU-RGBD 120 数据集，可使用以下脚本

```python
python gen_ntu_rgbd_raw.py --data-path your_raw_nturgbd120_skeleton_path --ignored-sample-path NTU_RGBD120_samples_with_missing_skeletons.txt --out-folder your_nturgbd120_output_path --task ntu120
```

## 转换其他第三方项目的骨骼标注

MMAction2 提供脚本以将其他第三方项目的骨骼标注转至 MMAction2 格式，如：

- BABEL: `babel2mma2.py`

**待办项**：

- [x] FineGYM
- [x] NTU60_XSub
- [x] NTU120_XSub
- [x] NTU60_XView
- [x] NTU120_XSet
- [x] UCF101
- [x] HMDB51
- [x] Kinetics
