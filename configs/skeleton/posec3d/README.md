# PoseC3D

[Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586)

<!-- [ALGORITHM] -->

## Abstract

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

## Results and Models

### FineGYM

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | Mean Top-1 | testing protocol | FLOPs | params |                 config                 |                 ckpt                 |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :--------: | :--------------: | :---: | :----: | :------------------------------------: | :----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |    93.5    |     10 clips     | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint_20220815-da338c58.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint/slowonly_r50_8xb16-u48-240e_gym-keypoint.log) |
|       uniform 48        |      limb      |  8   | SlowOnly-R50 |    93.6    |     10 clips     | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb_20220815-2e6e3c5c.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-limb/slowonly_r50_8xb16-u48-240e_gym-limb.log) |

### NTU60_XSub

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   93.6   |     10 clips     | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.log) |
|       uniform 48        |      limb      |  8   | SlowOnly-R50 |   93.5   |     10 clips     | 20.6G |  2.0M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb_20220815-af2f119a.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb/slowonly_r50_8xb16-u48-240e_ntu60-xsub-limb.log) |
|                         |     Fusion     |      |              |   94.0   |                  |       |        |                                         |                                       |                                      |

### UCF101

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   86.8   |     10 clips     | 14.6G |  3.1M  | [config](/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint_20220815-9972260d.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_ucf101-split1-keypoint.log) |

### HMDB51

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   69.6   |     10 clips     | 14.6G |  3.0M  | [config](/configs/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint_20220815-17eaa484.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint/slowonly_kinetics400-pretrained-r50_8xb16-u48-120e_hmdb51-split1-keypoint.log) |

# Kinetics400

| frame sampling strategy | pseudo heatmap | gpus |   backbone   | top1 acc | testing protocol | FLOPs | params |                 config                  |                 ckpt                  |                 log                  |
| :---------------------: | :------------: | :--: | :----------: | :------: | :--------------: | :---: | :----: | :-------------------------------------: | :-----------------------------------: | :----------------------------------: |
|       uniform 48        |    keypoint    |  8   | SlowOnly-R50 |   47.4   |     10 clips     | 19.1G |  3.2M  | [config](/configs/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint/slowonly_r50_8xb32-u48-240e_k400-keypoint_20230731-7f498b55.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb32-u48-240e_k400-keypoint/slowonly_r50_8xb32-u48-240e_k400-keypoint.log) |

You can follow the guide in [Preparing Skeleton Dataset](/tools/data/skeleton/README.md) to obtain skeleton annotations used in the above configs.

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train PoseC3D model on FineGYM dataset in a deterministic option.

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    --seed=0 --deterministic
```

For training with your custom dataset, you can refer to [Custom Dataset Training](/configs/skeleton/posec3d/custom_dataset_training.md).

For more details, you can refer to the **Training** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test PoseC3D model on FineGYM dataset.

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_gym-keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth
```

For more details, you can refer to the **Test** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

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
