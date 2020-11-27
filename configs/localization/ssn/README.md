# SSN

## Introduction
```
@InProceedings{Zhao_2017_ICCV,
author = {Zhao, Yue and Xiong, Yuanjun and Wang, Limin and Wu, Zhirong and Tang, Xiaoou and Lin, Dahua},
title = {Temporal Action Detection With Structured Segment Networks},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```

## Model Zoo

| config | gpus | backbone | pretrain | mAP@0.3 | mAP@0.4 | mAP@0.5 | reference mAP@0.3 | reference mAP@0.4 | reference mAP@0.5 | gpu_mem(M) | ckpt | log | json | refrence ckpt | refrence json
|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|:-:|:-:|:-:|---|:--:|:--:|
|[ssn_r50_450e_thumos14_rgb](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) |8| ResNet50 | ImageNet |29.37|22.15|15.69|[27.61](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|[21.28](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|[14.57](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|6352|[ckpt](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/ssn_r50_450e_thumos14_rgb_20201012-1920ab16.pth)| [log](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log)| [json](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log.json)| [ckpt](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/ssn_r50_450e_thumos14_rgb_ref_20201014-b6f48f68.pth)| [json](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/20201008_103258.log.json)|

- Notes:

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
e.g., lr=0.01 for 4 GPUs * 2 video/gpu and lr=0.08 for 16 GPUs * 4 video/gpu.
2. Since SSN utilizes different structured temporal pyramid pooling methods at training and testing, please refer to [ssn_r50_450e_thumos14_rgb_train](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) at training and [ssn_r50_450e_thumos14_rgb_test](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_test.py) at testing.
3. We evaluate the action detection performance of SSN, using action proposals of TAG. For more details on data preparation, you can refer to thumos14 TAG proposals in [Data Preparation](/docs/data_preparation.md).
4. The reference SSN in is evaluated with `ResNet50` backbone in MMAction, which is the same backbone with ours. Note that the original setting of MMAction SSN uses the `BNInception` backbone.

## Train

You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SSN model on thumos14 dataset.
```shell
python tools/train.py configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py
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
python tools/test.py configs/localization/ssn/ssn_r50_450e_thumos14_rgb_test.py checkpoints/SOME_CHECKPOINT.pth --eval mAP
```

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset) .
