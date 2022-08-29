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

#### NTU60_XSub

| config                                         | type  | gpus | backbone | Top-1 |                     ckpt                     |                     log                     |                     json                      |
| :--------------------------------------------- | :---: | :--: | :------: | :---: | :------------------------------------------: | :-----------------------------------------: | :-------------------------------------------: |
| [2sagcn_80e_ntu60_xsub_keypoint_3d](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py) | joint |  1   |   AGCN   | 86.06 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d-3bed61ba.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.json) |
| [2sagcn_80e_ntu60_xsub_bone_3d](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/ss-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py) | bone  |  2   |   AGCN   | 86.89 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d-278b8815.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.json) |

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train AGCN model on joint data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_keypoint_3d \
    --validate --seed 0 --deterministic
```

Example: train AGCN model on bone data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_bone_3d \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on joint data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out joint_result.pkl
```

Example: test AGCN model on bone data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out bone_result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

| config                                                  | pseudo heatmap | gpus |   backbone   | Mean Top-1 |                         ckpt                          |                         log                          |
| :------------------------------------------------------ | :------------: | :--: | :----------: | :--------: | :---------------------------------------------------: | :--------------------------------------------------: |
| [slowonly_r50_u48_240e_8xb16_gym_keypoint](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_keypoint.py) |    keypoint    |  8   | SlowOnly-R50 |    93.4    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_keypoint/slowonly_r50_u48_240e_8xb16_gym_keypoint_20220815-da338c58.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_keypoint/20220805_150313.log) |
| [slowonly_r50_u48_240e_8xb16_gym_limb](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_limb.py) |      limb      |  8   | SlowOnly-R50 |    93.7    | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_limb/slowonly_r50_u48_240e_8xb16_gym_limb_20220815-2e6e3c5c.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_limb/20220805_150302.log) |

#### NTU60_XSub

| config                                                   | pseudo heatmap | gpus |   backbone   | Top-1 |                          ckpt                           |                          log                           |
| :------------------------------------------------------- | :------------: | :--: | :----------: | :---: | :-----------------------------------------------------: | :----------------------------------------------------: |
| [slowonly_r50_u48_240e_8xb16_ntu60_xsub_keypoint](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_keypoint.py) |    keypoint    |  8   | SlowOnly-R50 | 93.6  | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_keypoint/slowonly_r50_u48_240e_8xb16_ntu60_xsub_keypoint_20220815-38db104b.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_keypoint/20220805_124409.log) |
| [slowonly_r50_u48_240e_8xb16_ntu60_xsub_limb](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_limb.py) |      limb      |  8   | SlowOnly-R50 | 93.5  | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_limb/slowonly_r50_u48_240e_8xb16_ntu60_xsub_limb_20220815-af2f119a.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_ntu60_xsub_limb/20220805_123439.log) |

#### UCF101

| config                                                   | pseudo heatmap | gpus |   backbone   | Top-1 |                          ckpt                           |                          log                           |
| :------------------------------------------------------- | :------------: | :--: | :----------: | :---: | :-----------------------------------------------------: | :----------------------------------------------------: |
| [slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_ucf101_split1_keypoint](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_ucf101_split1_keypoint.py) |    keypoint    |  8   | SlowOnly-R50 | 86.9  | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_ucf101_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_ucf101_split1_keypoint_20220815-9972260d.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_ucf101_split1_keypoint/20220805_130002.log) |

#### HMDB51

| config                                                   | pseudo heatmap | gpus |   backbone   | Top-1 |                          ckpt                           |                          log                           |
| :------------------------------------------------------- | :------------: | :--: | :----------: | :---: | :-----------------------------------------------------: | :----------------------------------------------------: |
| [slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_hmdb51_split1_keypoint](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_hmdb51_split1_keypoint.py) |    keypoint    |  8   | SlowOnly-R50 | 69.2  | [ckpt](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_hmdb51_split1_keypoint/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_hmdb51_split1_keypoint_20220815-17eaa484.pth) | [log](https://download.openmmlab.com/mmaction/v2.0/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_8xb16_hmdb51_split1_keypoint/20220805_143455.log) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 8 GPUs x 8 videos/gpu and lr=0.04 for 16 GPUs x 16 videos/gpu.
2. You can follow the guide in [Preparing Skeleton Dataset](https://github.com/open-mmlab/mmaction2/tree/master/tools/data/skeleton) to obtain skeleton annotations used in the above configs.

:::

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train PoseC3D model on FineGYM dataset in a deterministic option.

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_keypoint.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For training with your custom dataset, you can refer to [Custom Dataset Training](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/custom_dataset_training.md).

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test PoseC3D model on FineGYM dataset.

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_u48_240e_8xb16_gym_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

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

#### NTU60_XSub

| config                                        | keypoint | gpus | backbone | Top-1 |                    ckpt                     |                     log                     |                     json                     |
| :-------------------------------------------- | :------: | :--: | :------: | :---: | :-----------------------------------------: | :-----------------------------------------: | :------------------------------------------: |
| [stgcn_80e_ntu60_xsub_keypoint](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py) |    2d    |  2   |  STGCN   | 86.91 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.json) |
| [stgcn_80e_ntu60_xsub_keypoint_3d](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d.py) |    3d    |  1   |  STGCN   | 84.61 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.json) |

#### BABEL

| config                                     | gpus | backbone |   Top-1   | Mean Top-1 | Top-1 Official (AGCN) | Mean Top-1 Official (AGCN) |                   ckpt                   |                   log                    |
| ------------------------------------------ | :--: | :------: | :-------: | :--------: | :-------------------: | :------------------------: | :--------------------------------------: | :--------------------------------------: |
| [stgcn_80e_babel60](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_babel60.py) |  8   |  ST-GCN  | **42.39** | **28.28**  |         41.14         |           24.46            | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60-3d206418.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60.log) |
| [stgcn_80e_babel60_wfl](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_babel60_wfl.py) |  8   |  ST-GCN  | **40.31** |   29.79    |         33.41         |         **30.42**          | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60_wfl/stgcn_80e_babel60_wfl-1a9102d7.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60_wfl.log) |
| [stgcn_80e_babel120](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_babel120.py) |  8   |  ST-GCN  | **38.95** | **20.58**  |         38.41         |           17.56            | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120/stgcn_80e_babel120-e41eb6d7.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120.log) |
| [stgcn_80e_babel120_wfl](https://github.com/open-mmlab/mmaction2/tree/master/configs/skeleton/stgcn/stgcn_80e_babel120_wfl.py) |  8   |  ST-GCN  | **33.00** |   24.33    |         27.91         |        **26.17**\*         | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120_wfl/stgcn_80e_babel120_wfl-3f2c100d.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120_wfl.log) |

\* The number is copied from the [paper](https://arxiv.org/pdf/2106.09696.pdf), the performance of the [released checkpoints](https://github.com/abhinanda-punnakkal/BABEL/tree/main/action_recognition) for BABEL-120 is inferior.

### Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --work-dir work_dirs/stgcn_80e_ntu60_xsub_keypoint \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](getting_started.html#training-setting).

### Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test STGCN model on NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](getting_started.html#test-a-dataset).

### Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```
