# BSN
config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log
-|-|-|-|-|-|- | -
bsn_400x100_1x16_20e_activitynet_feature | None |74.91|66.31|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem]() [ckpt_pem]| [log_tem]() [log_pem]()

## Preparation
### Data
1. Put the rescaled feature data folder `csv_mean_100` under `$MMACTION/data/activitynet_feature_cuhk/`

    The raw feature data could be found at [here](https://github.com/wzmsltw/BSN-boundary-sensitive-network)

2. Put the annotaion files under `$MMACTION/data/ActivityNet`

    The annotation files could be found at [here]()

### Checkpoint
1. Put the `tem_best.pth.tar` and `pem_best.pth.tar` under `checkpoints/`

    The ckpts could be found at [here]() (TODO)

## Train
You can use the following commands to train a model.
```Bash
# TEM Train
python tools/train.py ${CONFIG_FILE} [optional arguments]
# Example: train BSN(TEM) on ActivityNet features dataset
python tools/train.py config/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py

# TEM Inference
# Note: This could not be evaluated.
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
# Example: Inference BSN(TEM) with trained model.
python tools/test.py config/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth

# PGM
python tools/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
# Example: Inference BSN(PGM)
python tools/bsn_proposal_generation.py config/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode train

# PEM Train
python tools/train.py ${CONFIG_FILE} [optional arguments]
# Example: train BSN(PEM) on PGM results.
python tools/train.py config/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py

# PEM Inference
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
# Example: Inference BSN(PEM) with evaluation metric 'AR@AN' and output the results.
# Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
python tools/test.py config/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py  checkpoints/SOME_CHECKPOINT.pth  --eval AR@AN --out results.json
```

## Test
```Bash
# TEM
# Note: This could not be evaluated.
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
# Example: Test a TEM model on ActivityNet dataset.
python tools/test.py config/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth

# PGM
python tools/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
# Example:
python tools/bsn_proposal_generation.py config/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode test

# PEM
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
# Example: Test a PEM model with with evaluation metric 'AR@AN' and output the results.
# Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
python tools/test.py config/localization/bsn/bsn_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
```
