# BSN

## Model Zoo

### ActivityNet feature

|config | gpus| pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|bsn_400x100_1x16_20e_activitynet_feature |1| None |74.65|66.45|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature_20200619-cd6accc3.pth) [ckpt_pem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature_20200619-6111891d.pth)| [log_tem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log) [log_pem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log)| [json_tem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log.json)  [json_pem](https://openmmlab.oss-accelerate.aliyuncs.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log.json)|

Notes:
1. The **gpus** indicates the number of gpu we used to get the checkpoint.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.

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
    ```python
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
    python tools/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
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
    python tools/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode train
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
    python tools/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
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
    python tools/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode test
    ```

3. Test a PEM model with with evaluation metric 'AR@AN' and output the results.
    ```shell
    python tools/test.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
    ```
For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).
