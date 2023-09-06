# DRN

[Dense Regression Network for Video Grounding](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zeng_Dense_Regression_Network_for_Video_Grounding_CVPR_2020_paper.pdf)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

We address the problem of video grounding from natural language queries. The key challenge in this task is that one training video might only contain a few annotated starting/ending frames that can be used as positive examples for model training. Most conventional approaches directly train a binary classifier using such imbalance data, thus achieving inferior results. The key idea of this paper is to use the distances between the frame within the ground truth and the starting (ending) frame as dense supervisions to improve the video grounding accuracy. Specifically, we design a novel dense regression network (DRN) to regress the distances from each frame to the starting (ending) frame of the video segment described by the query. We also propose a simple but effective IoU regression head module to explicitly consider the localization quality of the grounding results (i.e., the IoU between the predicted location and the ground truth). Experimental results show that our approach significantly outperforms state-of-the-arts on three datasets (i.e., Charades-STA, ActivityNet-Captions, and TACoS).

<!-- [IMAGE] -->

<div align=center>
<img src="https://github.com/open-mmlab/mmaction2/files/12532583/Fig1.pdf" width="800"/>
</div>

## Results and Models

### Charades STA C3D feature

| feature | gpus | pretrain | Recall@Top1(IoU=0.5) | Recall@Top5(IoU=0.5) |                      config                      |                      ckpt                       |                      log                       |
| :-----: | :--: | :------: | :------------------: | :------------------: | :----------------------------------------------: | :---------------------------------------------: | :--------------------------------------------: |
|   C3D   |  2   |   None   |        47.04         |        84.57         | [config](configs/localization/drn/drn_2xb16-4096-10e_c3d-feature_third.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/localization/drn/drn_2xb16-4096-10e_c3d-feature_20230809-ec0429a6.pth) | [log](https://download.openmmlab.com/mmaction/v1.0/drn_2xb16-4096-10e_c3d-feature.log) |

For more details on data preparation, you can refer to [Charades STA  Data Preparation](/tools/data/charades-sta/README.md).

## Train

The training of DRN has three stages. Following the official paper, the second and the third stage loads the best checkpoint from previous stage.

The first stage training:

```shell
bash tools/dist_train.sh configs/localization/drn/drn_2xb16-4096-10e_c3d-feature_first.py 2
```

The second stage training:

```shell
BEST_CKPT=work_dirs/drn_2xb16-4096-10e_c3d-feature_first/SOME.PTH
bash tools/dist_train.sh configs/localization/drn/drn_2xb16-4096-10e_c3d-feature_second.py 2 --cfg-options load_from=${BEST_CKPT}
```

The third stage training:

```shell
BEST_CKPT=work_dirs/drn_2xb16-4096-10e_c3d-feature_second/SOME.PTH
bash tools/dist_train.sh configs/localization/drn/drn_2xb16-4096-10e_c3d-feature_third.py 2 --cfg-options load_from=${BEST_CKPT}
```

## Test

Test DRN on Charades STA C3D feature:

```shell
python3 tools/test.py configs/localization/drn/drn_2xb16-4096-10e_c3d-feature_third.py CHECKPOINT.PTH
```

For more details, you can refer to the **Testing** part in the [Training and Test Tutorial](/docs/en/user_guides/train_test.md).

## Citation

```BibTeX
@inproceedings{DRN2020CVPR,
  author    = {Runhao, Zeng and Haoming, Xu and Wenbing, Huang and Peihao, Chen and Mingkui, Tan and Chuang Gan},
  title     = {Dense Regression Network for Video Grounding},
  booktitle = {CVPR},
  year      = {2020},
}
```

<!-- [DATASET] -->

```BibTeX
@inproceedings{gao2017tall,
  title={Tall: Temporal activity localization via language query},
  author={Gao, Jiyang and Sun, Chen and Yang, Zhenheng and Nevatia, Ram},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={5267--5275},
  year={2017}
}
```
