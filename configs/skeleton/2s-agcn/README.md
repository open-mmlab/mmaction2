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
|       uniform 100       |    joint     |  8   |   AGCN   |  88.60   |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221222-4c0ed77e.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       |     bone     |  8   |   AGCN   |  91.59   |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221222-293878b5.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | joint-motion |  8   |   AGCN   |  88.02   |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d_20221222-0c86e3a1.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|       uniform 100       | bone-motion  |  8   |   AGCN   |  88.82   |     10 clips     | 4.4G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d_20221222-87996f0d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-2d.log) |
|                         |  two-stream  |      |          |  91.95   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  92.34   |                  |       |        |                                           |                                         |                                        |

### NTU60_XSub_3D

| frame sampling strategy |   modality   | gpus | backbone | top1 acc | testing protocol | FLOPs | params |                  config                   |                  ckpt                   |                  log                   |
| :---------------------: | :----------: | :--: | :------: | :------: | :--------------: | :---: | :----: | :---------------------------------------: | :-------------------------------------: | :------------------------------------: |
|       uniform 100       |    joint     |  8   |   AGCN   |  88.26   |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d_20221222-24dabf78.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       |     bone     |  8   |   AGCN   |  89.22   |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d_20221222-abe70a7f.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | joint-motion |  8   |   AGCN   |  86.73   |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d_20221222-923cd3c3.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|       uniform 100       | bone-motion  |  8   |   AGCN   |  86.41   |     10 clips     | 6.5G  |  3.5M  | [config](/configs/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d_20221222-3d8f6f43.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d/2s-agcn_8xb16-bone-motion-u100-80e_ntu60-xsub-keypoint-3d.log) |
|                         |  two-stream  |      |          |  90.27   |                  |       |        |                                           |                                         |                                        |
|                         | four-stream  |      |          |  90.89   |                  |       |        |                                           |                                         |                                        |

1. The **gpus** indicates the number of gpus we used to get the checkpoint. If you want to use a different number of gpus or videos per gpu, the best way is to set `--auto-scale-lr` when calling `tools/train.py`, this parameter will auto-scale the learning rate according to the actual batch size, and the original batch size.
2. For two-stream fusion, we use **joint : bone = 1 : 1**. For four-stream fusion, we use **joint : joint-motion : bone : bone-motion = 2 : 1 : 2 : 1**. For more details about multi-stream fusion, please refer to this [tutorial](/docs/en/useful_tools.md#multi-stream-fusion).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train STGCN model on NTU60-2D dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --seed 0 --deterministic
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on NTU60-2D dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2s-agcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump result.pkl
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

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
