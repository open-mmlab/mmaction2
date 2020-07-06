# BMN

## Model Zoo

### ActivityNet feature

|config | pretrain | AR@100| AUC | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[bmn_400x100_9e_2x8_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py) | None |75.28|67.22|5420|3.27|[ckpt]()| [log]()|


For more details on data preparation, you can refer to [Prepaing Activitynet](/tools/data/activitynet/preparing_activitynet.md).

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
