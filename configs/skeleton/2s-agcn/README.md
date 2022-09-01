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

### NTU60_XSub

| frame sampling strategy | type  | gpus | backbone | top1 acc | testing protocol |                     config                      |                     ckpt                      |                     log                      |
| :---------------------: | :---: | :--: | :------: | :------: | :--------------: | :---------------------------------------------: | :-------------------------------------------: | :------------------------------------------: |
|       padding 300       | joint |  1   |   AGCN   |  86.06   |     1 clips      | [config](/configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d-3bed61ba.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d.log) |
|       padding 300       | bone  |  2   |   AGCN   |  86.89   |     1 clips      | [config](/configs/skeleton/2s-agcn/2sagcn_4xb16-80e-ntu60_xsub_bone-3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2sagcn_4xb16-80e-ntu60_xsub_bone-3d/2sagcn_4xb16-80e-ntu60_xsub_bone-3d-278b8815.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/2s-agcn/2sagcn_4xb16-80e-ntu60_xsub_bone-3d/2sagcn_4xb16-80e-ntu60_xsub_bone-3d.log) |

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train AGCN model on joint data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d.py \
    --work-dir work_dirs/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

Example: train AGCN model on bone data of NTU60 dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e-ntu60_xsub_bone-3d.py \
    --work-dir work_dirs/2sagcn_4xb16-80e-ntu60_xsub_bone-3d \
    --cfg-options randomness.seed=0 randomness.deterministic=True
```

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/4_train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test AGCN model on joint data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e_ntu60_xsub_keypoint-3d.py \
    checkpoints/SOME_CHECKPOINT.pth --dump joint_result.pkl
```

Example: test AGCN model on bone data of NTU60 dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_4xb16-80e-ntu60_xsub_bone-3d.py \
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
