# BMN

## Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an challenging and promising task which aims to locate temporal regions in real-world videos where action or event may occur. Current bottom-up proposal generation methods can generate proposals with precise boundary, but cannot efficiently generate adequately reliable confidence scores for retrieving proposals. To address these difficulties, we introduce the Boundary-Matching (BM) mechanism to evaluate confidence scores of densely distributed proposals, which denote a proposal as a matching pair of starting and ending boundaries and combine all densely distributed BM pairs into the BM confidence map. Based on BM mechanism, we propose an effective, efficient and end-to-end proposal generation method, named Boundary-Matching Network (BMN), which generates proposals with precise temporal boundaries as well as reliable confidence scores simultaneously. The two-branches of BMN are jointly trained in an unified framework. We conduct experiments on two challenging datasets: THUMOS-14 and ActivityNet-1.3, where BMN shows significant performance improvement with remarkable efficiency and generalizability. Further, combining with existing action classifier, BMN can achieve state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016479-2ca7e8b6-a17b-4a4c-b4c9-ae731935cd91.png" width="800"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{lin2019bmn,
  title={Bmn: Boundary-matching network for temporal action proposal generation},
  author={Lin, Tianwei and Liu, Xiao and Li, Xin and Ding, Errui and Wen, Shilei},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3889--3898},
  year={2019}
}
```

<!-- [DATASET] -->

```BibTeX
@article{zhao2017cuhk,
  title={Cuhk \& ethz \& siat submission to activitynet challenge 2017},
  author={Zhao, Y and Zhang, B and Wu, Z and Yang, S and Zhou, L and Yan, S and Wang, L and Xiong, Y and Lin, D and Qiao, Y and others},
  journal={arXiv preprint arXiv:1710.08011},
  volume={8},
  year={2017}
}
```

## Model Zoo

### ActivityNet feature

|config |feature | gpus | AR@100| AUC | AP@0.5 | AP@0.75 | AP@0.95 | mAP | gpu_mem(M) | iter time(s) | ckpt | log| json|
|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|---|:-:|:-:|---|
|[bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) |cuhk_mean_100 |2|75.28|67.22|42.47|31.31|9.92|30.34|5420|3.27|[ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature_20200619-42a3b111.pth)| [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log)| [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_9e_activitynet_feature/bmn_400x100_9e_activitynet_feature.log.json)|
| |mmaction_video |2|75.43|67.22|42.62|31.56|10.86|30.77|5420|3.27|[ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809-c9fd14d2.pth)| [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.log) | [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_video/bmn_400x100_2x8_9e_mmaction_video_20200809.json) |
| |mmaction_clip |2|75.35|67.38|43.08|32.19|10.73|31.15|5420|3.27|[ckpt](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809-10d803ce.pth)| [log](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.log) | [json](https://download.openmmlab.com/mmaction/localization/bmn/bmn_400x100_2x8_9e_mmaction_clip/bmn_400x100_2x8_9e_mmaction_clip_20200809.json) |
| [BMN-official](https://github.com/JJBOY/BMN-Boundary-Matching-Network) (for reference)* |cuhk_mean_100 |-|75.27|67.49|42.22|30.98|9.22|30.00|-|-|-| - | - |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk), mmaction_video and mmaction_clip denote feature extracted by mmaction, with video-level activitynet finetuned model or clip-level activitynet finetuned model respectively.
3. We evaluate the action detection performance of BMN, using  [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) submission for ActivityNet2017 Untrimmed Video Classification Track to assign label for each action proposal.

:::

*We train BMN with the [official repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network), evaluate its proposal generation and action detection performance with [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) for label assigning.

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

You can also test the action detection performance of the model, with [anet_cuhk_2017](https://download.openmmlab.com/mmaction/localization/cuhk_anet17_pred.json) prediction file and generated proposal file (`results.json` in last command).

```shell
python tools/analysis/report_map.py --proposal path/to/proposal_file
```

:::{note}

1. (Optional) You can use the following command to generate a formatted proposal file, which will be fed into the action classifier (Currently supports SSN and P-GCN, not including TSN, I3D etc.) to get the classification result of proposals.

    ```shell
    python tools/data/activitynet/convert_proposal_format.py
    ```

:::

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
