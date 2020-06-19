# BMN
|config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|- | -|
|bmn_400x100_9e_2x8_activitynet_feature | None |75.28|67.22|5420|3.27|[ckpt]()| [log]()|

### Data
1. Put the rescaled feature data folder `csv_mean_100` under `$MMACTION/data/activitynet_feature_cuhk/`

    The raw feature data could be found at [here](https://github.com/wzmsltw/BSN-boundary-sensitive-network)

2. Put the annotaion files under `$MMACTION/data/ActivityNet`

    The annotation files could be found at [here]()

### Checkpoint
1. Put the `tem_best.pth.tar` and `pem_best.pth.tar` under `checkpoints/`

    The ckpts could be found at [here]()

## Train
You can use the following command to train a model.
```Bash
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Example: train BMN on ActivityNet features dataset
python tools/train.py config/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py
```

## Test
You can use the following command to test a model.
```Bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Example: test BMN on ActivityNet feature dataset
# Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
python tools/test.py  config/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
```
