# BMN

## Introduction
```
@inproceedings{lin2019bmn,
  title={Bmn: Boundary-matching network for temporal action proposal generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3889--3898},
  year={2019}
}
```

## Model Zoo

### ActivityNet feature

|config |feature | gpus | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
|[bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) |cuhk_mean_100 |2| None |75.28|67.22|5420|3.27|[ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature_20200619-42a3b111.pth)| [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log)| [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log.json)|
| |mmaction_video |2| None |75.43|67.22|5420|3.27|[ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809-c9fd14d2.pth)| [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.log) | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.json) |
| |mmaction_clip |2| None |75.35|67.38|5420|3.27|[ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809-10d803ce.pth)| [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.log) | [json](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.json) |

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk), mmaction_video and mmaction_clip denote feature extracted by mmaction, with video-level activitynet finetuned model or clip-level activitynet finetuned model respectively.

For more details on data preparation, you can refer to ActivityNet feature in [Data Preparation](/docs/data_preparation.md).

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train BMN model on ActivityNet features dataset.
```shell
python tools/train.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py
```
For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting) .

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test BMN on ActivityNet feature dataset.
```shell
# Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
python tools/test.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
```
For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
