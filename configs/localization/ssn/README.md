# SSN

[Temporal Action Detection With Structured Segment Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhao_Temporal_Action_Detection_ICCV_2017_paper.html)

<!-- [ALGORITHM] -->

## Abstract

<!-- [ABSTRACT] -->

Detecting actions in untrimmed videos is an important yet challenging task. In this paper, we present the structured segment network (SSN), a novel framework which models the temporal structure of each action instance via a structured temporal pyramid. On top of the pyramid, we further introduce a decomposed discriminative model comprising two classifiers, respectively for classifying actions and determining completeness. This allows the framework to effectively distinguish positive proposals from background or incomplete ones, thus leading to both accurate recognition and localization. These components are integrated into a unified network that can be efficiently trained in an end-to-end fashion. Additionally, a simple yet effective temporal action proposal scheme, dubbed temporal actionness grouping (TAG) is devised to generate high quality action proposals. On two challenging benchmarks, THUMOS14 and ActivityNet, our method remarkably outperforms previous state-of-the-art methods, demonstrating superior accuracy and strong adaptivity in handling actions with various temporal structures.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/143016899-017893d3-a907-4487-90a2-cb884088266c.png" width="800"/>
</div>

## Results and Models

| config | gpus | backbone | pretrain | mAP@0.3 | mAP@0.4 | mAP@0.5 | reference mAP@0.3 | reference mAP@0.4 | reference mAP@0.5 | gpu_mem(M) | ckpt | log | json | reference ckpt | reference json
|:-:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|:-:|:-:|:-:|---|:--:|:--:|
|[ssn_r50_450e_thumos14_rgb](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) |8| ResNet50 | ImageNet |29.37|22.15|15.69|[27.61](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|[21.28](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|[14.57](https://github.com/open-mmlab/mmaction/tree/c7e3b7c11fb94131be9b48a8e3d510589addc3ce#Get%20started)|6352|[ckpt](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/ssn_r50_450e_thumos14_rgb_20201012-1920ab16.pth)| [log](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log)| [json](https://download.openmmlab.com/mmaction/localization/ssn/ssn_r50_450e_thumos14_rgb/20201005_144656.log.json)| [ckpt](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/ssn_r50_450e_thumos14_rgb_ref_20201014-b6f48f68.pth)| [json](https://download.openmmlab.com/mmaction/localization/ssn/mmaction_reference/ssn_r50_450e_thumos14_rgb_ref/20201008_103258.log.json)|

:::{note}

1. The **gpus** indicates the number of gpu we used to get the checkpoint.
   According to the [Linear Scaling Rule](https://arxiv.org/abs/1706.02677), you may set the learning rate proportional to the batch size if you use different GPUs or videos per GPU,
   e.g., lr=0.01 for 4 GPUs x 2 video/gpu and lr=0.08 for 16 GPUs x 4 video/gpu.
2. Since SSN utilizes different structured temporal pyramid pooling methods at training and testing, please refer to [ssn_r50_450e_thumos14_rgb_train](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py) at training and [ssn_r50_450e_thumos14_rgb_test](/configs/localization/ssn/ssn_r50_450e_thumos14_rgb_test.py) at testing.
3. We evaluate the action detection performance of SSN, using action proposals of TAG. For more details on data preparation, you can refer to thumos14 TAG proposals in [Data Preparation](/docs/data_preparation.md).
4. The reference SSN in is evaluated with `ResNet50` backbone in MMAction, which is the same backbone with ours. Note that the original setting of MMAction SSN uses the `BNInception` backbone.

:::

## Train

You can use the following command to train a model.

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train SSN model on thumos14 dataset.

```shell
python tools/train.py configs/localization/ssn/ssn_r50_450e_thumos14_rgb_train.py
```

For more details and optional arguments infos, you can refer to **Training setting** part in [getting_started](/docs/getting_started.md#training-setting).

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

For more details and optional arguments infos, you can refer to **Test a dataset** part in [getting_started](/docs/getting_started.md#test-a-dataset).

## Citation

```BibTeX
@InProceedings{Zhao_2017_ICCV,
author = {Zhao, Yue and Xiong, Yuanjun and Wang, Limin and Wu, Zhirong and Tang, Xiaoou and Lin, Dahua},
title = {Temporal Action Detection With Structured Segment Networks},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2017}
}
```
