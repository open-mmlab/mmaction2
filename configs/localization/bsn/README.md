# BSN

## Abstract

<!-- [ABSTRACT] -->

Temporal action proposal generation is an important yet challenging problem, since temporal proposals with rich action content are indispensable for analysing real-world videos with long duration and high proportion irrelevant content. This problem requires methods not only generating proposals with precise temporal boundaries, but also retrieving proposals to cover truth action instances with high recall and high overlap using relatively fewer proposals. To address these difficulties, we introduce an effective proposal generation method, named Boundary-Sensitive Network (BSN), which adopts "local to global" fashion. Locally, BSN first locates temporal boundaries with high probabilities, then directly combines these boundaries as proposals. Globally, with Boundary-Sensitive Proposal feature, BSN retrieves proposals by evaluating the confidence of whether a proposal contains an action within its region. We conduct experiments on two challenging datasets: ActivityNet-1.3 and THUMOS14, where BSN outperforms other state-of-the-art temporal action proposal generation methods with high recall and high temporal precision. Finally, further experiments demonstrate that by combining existing action classifiers, our method significantly improves the state-of-the-art temporal action detection performance.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016692-69efafbd-cec6-47f1-af45-371d0ff78a97.png" width="800"/>
</div>

## Citation

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{lin2018bsn,
  title={Bsn: Boundary sensitive network for temporal action proposal generation},
  author={Lin, Tianwei and Zhao, Xu and Su, Haisheng and Wang, Chongjing and Yang, Ming},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={3--19},
  year={2018}
}
```

## Model Zoo

### ActivityNet feature

|config |feature | gpus| pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
|bsn_400x100_1x16_20e_activitynet_feature |cuhk_mean_100 |1| None |74.66|66.45|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature_20200619-cd6accc3.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature_20210203-1c27763d.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log)| [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log.json)  [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log.json)|
| |mmaction_video |1| None |74.93|66.74|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809-ad6ec626.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809-aa861b26.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809.log) | [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809.json) [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809.json) |
| |mmaction_clip |1| None |75.19|66.81|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809-0a563554.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809-e32f61e6.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809.log) | [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809.json) [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809.json) |

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. For feature column, cuhk_mean_100 denotes the widely used cuhk activitynet feature extracted by [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk), mmaction_video and mmaction_clip denote feature extracted by mmaction, with video-level activitynet finetuned model or clip-level activitynet finetuned model respectively.

:::

For more details on data preparation, you can refer to ActivityNet feature in [Data Preparation](/docs/data_preparation.md).

## Train

You can use the following commands to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Examples:

1. train BSN(TEM) on ActivityNet features dataset.

    ```shell
    python tools/train.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py
    ```

2. train BSN(PEM) on PGM results.

    ```shell
    python tools/train.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py
    ```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

## Inference

You can use the following commands to inference a model.

1. For TEM Inference

    ```shell
    # Note: This could not be evaluated.
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

2. For PGM Inference

    ```shell
    python tools/misc/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
    ```

3. For PEM Inference

    ```shell
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

Examples:

1. Inference BSN(TEM) with pretrained model.

    ```shell
    python tools/test.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth
    ```

2. Inference BSN(PGM) with pretrained model.

    ```shell
    python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode train
    ```

3. Inference BSN(PEM) with evaluation metric 'AR@AN' and output the results.

    ```shell
    # Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
    python tools/test.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py  checkpoints/SOME_CHECKPOINT.pth  --eval AR@AN --out results.json
    ```

## Test

You can use the following commands to test a model.

1. TEM

    ```shell
    # Note: This could not be evaluated.
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

2. PGM

    ```shell
    python tools/misc/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
    ```

3. PEM

    ```shell
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

Examples:

1. Test a TEM model on ActivityNet dataset.

    ```shell
    python tools/test.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth
    ```

2. Test a PGM model on ActivityNet dataset.

    ```shell
    python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode test
    ```

3. Test a PEM model with with evaluation metric 'AR@AN' and output the results.

    ```shell
    python tools/test.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
    ```

:::{note}

1. (Optional) You can use the following command to generate a formatted proposal file, which will be fed into the action classifier (Currently supports only SSN and P-GCN, not including TSN, I3D etc.) to get the classification result of proposals.

    ```shell
    python tools/data/activitynet/convert_proposal_format.py
    ```

:::

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
