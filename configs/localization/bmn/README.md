# BMN

## Model Zoo

### ActivityNet feature

|config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) | None |75.28|67.22|5420|3.27|[ckpt]()| [log]()|

## Data
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

## Checkpoint
Put the `tem_best.pth.tar` and `pem_best.pth.tar` under `checkpoints/`.
The ckpts could be found at [here]().

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train BMN model on ActivityNet features dataset.
```shell
python tools/train.py configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py
```
For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](../../../docs/getting_started.md).

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
For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](../../../docs/getting_started.md).
