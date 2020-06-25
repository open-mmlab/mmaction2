# BSN

|config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|- | -|
|bsn_400x100_1x16_20e_activitynet_feature | None |74.65|66.45|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem]() [ckpt_pem]| [log_tem]() [log_pem]()|

## Preparation
### Data
1. Put the rescaled feature data folder `csv_mean_100` under `$MMACTION/data/activitynet_feature_cuhk/`.

    The raw feature data could be found at [here](https://github.com/wzmsltw/BSN-boundary-sensitive-network).

2. Put the annotaion files under `$MMACTION/data/ActivityNet`.

    The annotation files could be found at [here]().

3. Finally, make sure your folder structure same with the tree structure below.
If your folder structure is different, you can also change the corresponding paths in config files.
```
mmaction
├── mmaction
├── tools
├── config
├── data
│   ├── activitynet_feature_cuhk
│   │   ├── csv_mean_100
│   ├── ActivityNet
│   │   ├── anet_anno_train.json
│   │   ├── anet_anno_val.json
│   │   ├── anet_anno_test.json
...
```

### Checkpoint
1. Put the `tem_best.pth.tar` and `pem_best.pth.tar` under `checkpoints/`.

    The ckpts could be found at [here]() (TODO).

## Train
You can use the following commands to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Examples:

1. train BSN(TEM) on ActivityNet features dataset.
    ```shell
    python tools/train.py configs/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py
    ```

2. train BSN(PEM) on PGM results.
    ```python
    python tools/train.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py
    ```

For more details and optional arguments infos, you can refer to **Training setting** part in [GETTING_START](../../../docs/GETTING_STARTED.md).

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
    python tools/test.py configs/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py  checkpoints/SOME_CHECKPOINT.pth  --eval AR@AN --out results.json
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
    python tools/test.py configs/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
    ```
For more details and optional arguments infos, you can refer to **Test a dataset** part in [GETTING_START](../../../docs/GETTING_STARTED.md).
