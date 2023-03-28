# Skeleton-based Action Recognition Models

## AGCN

[Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

In skeleton-based action recognition, graph convolutional networks (GCNs), which model the human body skeletons as spatiotemporal graphs, have achieved remarkable performance. However, in existing GCN-based methods, the topology of the graph is set manually, and it is fixed over all layers and input samples. This may not be optimal for the hierarchical GCN and diverse samples in action recognition tasks. In addition, the second-order information (the lengths and directions of bones) of the skeleton data, which is naturally more informative and discriminative for action recognition, is rarely investigated in existing methods. In this work, we propose a novel two-stream adaptive graph convolutional network (2s-AGCN) for skeleton-based action recognition. The topology of the graph in our model can be either uniformly or individually learned by the BP algorithm in an end-to-end manner. This data-driven method increases the flexibility of the model for graph construction and brings more generality to adapt to various data samples. Moreover, a two-stream framework is proposed to model both the first-order and the second-order information simultaneously, which shows notable improvement for the recognition accuracy. Extensive experiments on the two large-scale datasets, NTU-RGBD and Kinetics-Skeleton, demonstrate that the performance of our model exceeds the state-of-the-art with a significant margin.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/30782254/143212681-a676d7a0-e92b-4a8a-ad8c-c5826eb58019.png" width="800"/>
</div>

### Results and Models

#### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |   AGCN   |  88.60   |     10 clips     | 4.4G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |   AGCN   |  91.59   |     10 clips     | 4.4G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221222-293878b5.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |   AGCN   |  88.02   |     10 clips     | 4.4G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221222-0c86e3a1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |   AGCN   |  88.82   |     10 clips     | 4.4G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221222-87996f0d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  91.95   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.34   |                  |       |        |                                           |                                         |                                        |

#### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |   AGCN   |  88.26   |     10 clips     | 6.5G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221222-24dabf78.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |   AGCN   |  89.22   |     10 clips     | 6.5G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221222-abe70a7f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |   AGCN   |  86.73   |     10 clips     | 6.5G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221222-923cd3c3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |   AGCN   |  86.41   |     10 clips     | 6.5G  |  3.5M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221222-3d8f6f43.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  90.27   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  90.89   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](en/user_guides/useful_tools.md##multi-stream-fusion).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{shi2019two,
  title={Two-stream adaptive graph convolutional networks for skeleton-based action recognition},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12026--12035},
  year={2019}
}
```

## PoseC3D

[Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Human skeleton, as a compact representation of human action, has received increasing attention in recent years. Many skeleton-based action recognition methods adopt graph convolutional networks (GCN) to extract features on top of human skeletons. Despite the positive results shown in previous works, GCN-based methods are subject to limitations in robustness, interoperability, and scalability. In this work, we propose PoseC3D, a new approach to skeleton-based action recognition, which relies on a 3D heatmap stack instead of a graph sequence as the base representation of human skeletons. Compared to GCN-based methods, PoseC3D is more effective in learning spatiotemporal features, more robust against pose estimation noises, and generalizes better in cross-dataset settings. Also, PoseC3D can handle multiple-person scenarios without additional computation cost, and its features can be easily integrated with other modalities at early fusion stages, which provides a great design space to further boost the performance. On four challenging datasets, PoseC3D consistently obtains superior performance, when used alone on skeletons and in combination with the RGB modality.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995620-21b5536c-8cda-48cd-9cb9-50b70cab7a89.png" width="800"/>
</div>

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

### Results and Models

#### FineGYM

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | Mean Top-1 | testing protocol | FLOPs | params |                 config                 |                 ckpt                 |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :--------: | :--------------: | :---: | :----: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |    93.5    |     10 clips     | 20.6G |  2.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint_20220815-da338c58.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint.log) |
|       uniform 48        |      limb      |  8   | SlowOnly-R50 |    93.6    |     10 clips     | 20.6G |  2.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb.log) |

#### NTU60_XSub

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   93.6   |     10 clips     | 20.6G |  2.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.log) |
|       uniform 48        |      limb      |  8   | SlowOnly-R50 |   93.5   |     10 clips     | 20.6G |  2.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb_20220815-af2f119a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.log) |
|                         |     Fusion     |      |              |   94.0   |                  |       |        |                                         |                                       |                                      |

#### UCF101

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   86.8   |     10 clips     | 14.6G |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint_20220815-9972260d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.log) |

#### HMDB51

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   69.6   |     10 clips     | 14.6G |  3.0M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.log) |

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 8 GPUs x 8 videos/gpu and lr=0.04 for 16 GPUs x 16 videos/gpu.
2. You can follow the guide in [Preparing Skeleton Dataset](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/skeleton/README.md) to obtain skeleton annotations used in the above configs.

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train PoseC3D model on FineGYM dataset in a deterministic option.

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For training with your custom dataset, you can refer to [Custom Dataset Training](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/custom_dataset_training.md).

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test PoseC3D model on FineGYM dataset.

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

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

## STGCN++

[PYSKL: Towards Good Practices for Skeleton Action Recognition](https://arxiv.org/abs/2205.09443)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

We present PYSKL: an open-source toolbox for skeleton-based action recognition based on PyTorch. The toolbox supports a wide variety of skeleton action recognition algorithms, including approaches based on GCN and CNN. In contrast to existing open-source skeleton action recognition projects that include only one or two algorithms, PYSKL implements six different algorithms under a unified framework with both the latest and original good practices to ease the comparison of efficacy and efficiency. We also provide an original GCN-based skeleton action recognition model named ST-GCN++, which achieves competitive recognition performance without any complicated attention schemes, serving as a strong baseline. Meanwhile, PYSKL supports the training and testing of nine skeleton-based action recognition benchmarks and achieves state-of-the-art recognition performance on eight of them. To facilitate future research on skeleton action recognition, we also provide a large number of trained models and detailed benchmark results to give some insights. PYSKL is released at this https URL and is actively maintained. We will update this report when we add new features or benchmarks. The current version corresponds to PYSKL v0.2.

### Results and Models

#### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.29   |     10 clips     | 1.95G | 1.39M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221228-86e1e77a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  92.30   |     10 clips     | 1.95G | 1.39M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221228-cd11a691.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  87.30   |     10 clips     | 1.95G | 1.39M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-19a34aba.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  88.76   |     10 clips     | 1.95G | 1.39M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221228-c02a0749.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  92.61   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.77   |                  |       |        |                                           |                                         |                                        |

#### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | STGCN++  |  89.14   |     10 clips     | 2.96G |  1.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221230-4e455ce3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   | STGCN++  |  90.21   |     10 clips     | 2.96G |  1.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221230-7f356072.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   | STGCN++  |  86.67   |     10 clips     | 2.96G |  1.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-650de5cc.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   | STGCN++  |  87.45   |     10 clips     | 2.96G |  1.4M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221230-b00440d2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcnpp/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcnpp_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  91.39   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  91.87   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](en/user_guides/useful_tools.md##multi-stream-fusion).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN++ model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN++ model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@misc{duan2022PYSKL,
  url = {https://arxiv.org/abs/2205.09443},
  author = {Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  title = {PYSKL: Towards Good Practices for Skeleton Action Recognition},
  publisher = {arXiv},
  year = {2022}
}
```

## STGCN

[Spatial temporal graph convolutional networks for skeleton-based action recognition](https://ojs.aaai.org/index.php/AAAI/article/view/12328)

<!-- [ALGORITHM] -->

### Abstract

<!-- [ABSTRACT] -->

Dynamics of human body skeletons convey significant information for human action recognition. Conventional approaches for modeling skeletons usually rely on hand-crafted parts or traversal rules, thus resulting in limited expressive power and difficulties of generalization. In this work, we propose a novel model of dynamic skeletons called Spatial-Temporal Graph Convolutional Networks (ST-GCN), which moves beyond the limitations of previous methods by automatically learning both the spatial and temporal patterns from data. This formulation not only leads to greater expressive power but also stronger generalization capability. On two large datasets, Kinetics and NTU-RGBD, it achieves substantial improvements over mainstream methods.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/142995893-d6618728-072c-46e1-b276-9b88cf21a01c.png" width="800"/>
</div>

### Results and Models

#### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  88.95   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  91.69   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  86.90   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-f18eb408.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  87.86   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-99c60e2d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  92.12   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.34   |                  |       |        |                                           |                                         |                                        |

#### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  88.11   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221129-850308e1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  88.76   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221129-9c8d2970.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  86.06   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-927648ea.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  85.49   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-593162ca.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  90.14   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  90.39   |                  |       |        |                                           |                                         |                                        |

#### NTU120_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  83.19   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d_20221129-612416c6.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  83.36   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d_20221129-131e63c3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  78.87   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d_20221129-7cb38ec2.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  79.55   |     10 clips     | 3.8G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d_20221129-f5b19892.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  84.84   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  85.23   |                  |       |        |                                           |                                         |                                        |

#### NTU120_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |  STGCN   |  82.15   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d_20221129-0484f579.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |  STGCN   |  84.28   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d_20221129-bc007510.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |  STGCN   |  78.93   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d_20221129-5d54f525.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu120-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |  STGCN   |  80.02   |     10 clips     | 5.7G  |  3.1M  | [config](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d_20221129-3cb0e4e1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu120-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  85.68   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  86.19   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](en/user_guides/useful_tools.md##multi-stream-fusion).

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](en/user_guides/4_train_test.md).

### Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```
