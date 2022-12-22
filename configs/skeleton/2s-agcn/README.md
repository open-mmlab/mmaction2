# AGCN

[Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

In skeleton-based action recognition, graph convolutional networks (GCNs), which model the human body skeletons as spatiotemporal graphs, have achieved remarkable performance. However, in existing GCN-based methods, the topology of the graph is set manually, and it is fixed over all layers and input samples. This may not be optimal for the hierarchical GCN and diverse samples in action recognition tasks. In addition, the second-order information (the lengths and directions of bones) of the skeleton data, which is naturally more informative and discriminative for action recognition, is rarely investigated in existing methods. In this work, we propose a novel two-stream adaptive graph convolutional network (2s-AGCN) for skeleton-based action recognition. The topology of the graph in our model can be either uniformly or individually learned by the BP algorithm in an end-to-end manner. This data-driven method increases the flexibility of the model for graph construction and brings more generality to adapt to various data samples. Moreover, a two-stream framework is proposed to model both the first-order and the second-order information simultaneously, which shows notable improvement for the recognition accuracy. Extensive experiments on the two large-scale datasets, NTU-RGBD and Kinetics-Skeleton, demonstrate that the performance of our model exceeds the state-of-the-art with a significant margin.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/30782254/143212681-a676d7a0-e92b-4a8a-ad8c-c5826eb58019.png" width="800"/>
</div>

## Results and Models

### NTU60_XSub_2D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | AGCN  |          |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   | AGCN  |          |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   | AGCN  |          |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-f18eb408.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   | AGCN  |          |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221129-99c60e2d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |          |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |          |                  |       |        |                                           |                                         |                                        |

### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   | AGCN  |          |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221129-850308e1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   | AGCN  |          |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221129-9c8d2970.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   | AGCN  |          |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-927648ea.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   | AGCN  |          |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221129-593162ca.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/stgcn/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/stgcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |          |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |          |                  |       |        |                                           |                                         |                                        |

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train AGCN model on **joint data** of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60-xsub-keypoint-3d.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

Example: train AGCN model on **bone data** of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60-xsub-bone-3d.py \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on **joint data** of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60-xsub-keypoint-3d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump joint_result.pkl
```

Example: test AGCN model on **bone data** of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60-xsub-bone-3d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump bone_result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Citation

```BibTeX
@inproceedings{shi2019two,
  title={Two-stream adaptive graph convolutional networks for skeleton-based action recognition},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12026--12035},
  year={2019}
}
```
