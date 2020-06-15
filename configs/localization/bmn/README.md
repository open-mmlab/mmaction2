# BMN
config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log
-|-|-|-|-|-|- | -
bmn_400x100_9e_activitynet_feature | None |-|-|-|-|[ckpt]()| [log]()

### Data
1. Put the rescaled feature data folder `csv_mean_100` under `$MMACTION/data/activitynet_feature_cuhk/`

    The raw feature data could be found at [here](https://github.com/wzmsltw/BSN-boundary-sensitive-network)

2. Put the annotaion files under `$MMACTION/data/ActivityNet`

    The annotation files could be found at [here]() (TODO)

### Checkpoint
1. Put the `tem_best.pth.tar` and `pem_best.pth.tar` under `checkpoints/`

    The ckpts could be found at [here]() (TODO)

## Train
You can use the following command to train a model.
```Bash
python tools/train.py ${CONFIG_FILE} [optional arguments]

# Example: train BMN on ActivityNet features dataset
python tools/train.py config/localization/bmn_feature_100_activitynet.py
```

## Test
You can use the following command to test a model.
```Bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]

# Example: test BMN on ActivityNet feature dataset
# Note: If evaluated, then please make sure the annotation file for test data contains groundtruth.
python tools/test.py  config/localization/bmn_feature_100_activitynet.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
```
