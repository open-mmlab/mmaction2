# PoseC3D

## Introduction

<!-- [ALGORITHM] -->

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

## Model Zoo

### FineGYM

|config |pseudo heatmap | gpus | backbone | Mean Top-1 | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
|[slowonly_r50_u48_240e_gym_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py) |keypoint |8 x 2| SlowOnly-R50 |93.7 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint-b07a98a0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint/slowonly_r50_u48_240e_gym_keypoint.json) |
|[slowonly_r50_u48_240e_gym_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb.py) |limb |8 x 2| SlowOnly-R50 |94.0 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb-c0d7b482.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_gym_limb/slowonly_r50_u48_240e_gym_limb.json) |
|Fusion | ||  |94.3 |  | | |

### NTU60_XSub

| config                                                       | pseudo heatmap | gpus  |   backbone   | Top-1 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :---: | :----------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_u48_240e_ntu60_xsub_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py) |    keypoint    | 8 x 2 | SlowOnly-R50 | 93.7  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint/slowonly_r50_u48_240e_ntu60_xsub_keypoint.json) |
| [slowonly_r50_u48_240e_ntu60_xsub_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb.py) |      limb      | 8 x 2 | SlowOnly-R50 | 93.4  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb-1d69006a.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_limb/slowonly_r50_u48_240e_ntu60_xsub_limb.json) |
| Fusion                                                       |                |       |              | 94.1  |                                                              |                                                              |                                                              |

### NTU120_XSub

| config                                                       | pseudo heatmap | gpus  |   backbone   | Top-1 |                             ckpt                             |                             log                              |                             json                             |
| :----------------------------------------------------------- | :------------: | :---: | :----------: | :---: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [slowonly_r50_u48_240e_ntu120_xsub_keypoint](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py) |    keypoint    | 8 x 2 | SlowOnly-R50 | 86.3  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint.json) |
| [slowonly_r50_u48_240e_ntu120_xsub_limb](/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb.py) |      limb      | 8 x 2 | SlowOnly-R50 | 85.7  | [ckpt](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb-803c2317.pth?) | [log](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb.log) | [json](https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_limb/slowonly_r50_u48_240e_ntu120_xsub_limb.json) |
| Fusion                                                       |                |       |              | 86.9  |                                                              |                                                              |                                                              |

Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint. It is noteworthy that the configs we provide are used for 8 gpus as default.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 8 GPUs x 8 videos/gpu and lr=0.04 for 16 GPUs x 16 videos/gpu.
2. The values in columns named after "reference" are the results got by testing on our dataset, using the checkpoints provided by the author with same model settings. The checkpoints for reference repo can be downloaded [here](https://drive.google.com/drive/folders/1sFfmP3yrfc7IzRshEELOby7-aEoymIFL?usp=sharing).

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train PoseC3D model on FineGYM dataset in a deterministic option with periodic validation.

```shell
python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py \
    --work-dir work_dirs/slowonly_r50_u48_240e_gym_keypoint \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Test

You can use the following command to test a model.

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test PoseC3D model on FineGYM dataset and dump the result to a pickle file.

```shell
python tools/test.py configs/skeleton/posec3d/slowonly_r50_u48_240e_gym_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

For more details, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
