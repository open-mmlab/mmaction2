# BSN

## 简介

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

## 模型库

### ActivityNet feature

|配置文件 |特征 | GPU 数量| 预训练 | AR@100| AUC | GPU 显存占用 (M) | 迭代时间 (s) | ckpt | log| json|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:-:|
|bsn_400x100_1x16_20e_activitynet_feature |cuhk_mean_100 |1| None |74.66|66.45|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature_20200619-cd6accc3.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature_20210203-1c27763d.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log)| [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature/bsn_tem_400x100_1x16_20e_activitynet_feature.log.json)  [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature/bsn_pem_400x100_1x16_20e_activitynet_feature.log.json)|
| |mmaction_video |1| None |74.93|66.74|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809-ad6ec626.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809-aa861b26.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809.log) | [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_video/bsn_tem_400x100_1x16_20e_mmaction_video_20200809.json) [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_video/bsn_pem_400x100_1x16_20e_mmaction_video_20200809.json) |
| |mmaction_clip |1| None |75.19|66.81|41(TEM)+25(PEM)|0.074(TEM)+0.036(PEM)|[ckpt_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809-0a563554.pth) [ckpt_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809-e32f61e6.pth)| [log_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809.log) [log_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809.log) | [json_tem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809.json) [json_pem](https://download.openmmlab.com/mmaction/localization/bsn/bsn_pem_400x100_1x16_20e_mmaction_clip/bsn_pem_400x100_1x16_20e_mmaction_clip_20200809.json) |

注：

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 对于 **特征** 这一列，`cuhk_mean_100` 表示所使用的特征为利用 [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) 代码库抽取的，被广泛利用的 CUHK ActivityNet 特征，
   `mmaction_video` 和 `mmaction_clip` 分布表示所使用的特征为利用 MMAction 抽取的，视频级别 ActivityNet 预训练模型的特征；视频片段级别 ActivityNet 预训练模型的特征。

对于数据集准备的细节，用户可参考 [数据集准备文档](/docs_zh_CN/data_preparation.md) 中的 ActivityNet 特征部分。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：

1. 在 ActivityNet 特征上训练 BSN(TEM) 模型。

    ```shell
    python tools/train.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py
    ```

2. 基于 PGM 的结果训练 BSN(PEM)。

    ```shell
    python tools/train.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py
    ```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何进行推理

用户可以使用以下指令进行模型推理。

1. 推理 TEM 模型。

    ```shell
    # Note: This could not be evaluated.
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

2. 推理 PGM 模型

    ```shell
    python tools/misc/bsn_proposal_generation.py ${CONFIG_FILE} [--mode ${MODE}]
    ```

3. 推理 PEM 模型

    ```shell
    python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
    ```

例如

1. 利用预训练模型进行 BSN(TEM) 模型的推理。

    ```shell
    python tools/test.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth
    ```

2. 利用预训练模型进行 BSN(PGM) 模型的推理

    ```shell
    python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode train
    ```

3. 推理 BSN(PEM) 模型，并计算 'AR@AN' 指标，输出结果文件。

    ```shell
    # 注：如果需要进行指标验证，需确测试数据的保标注文件包含真实标签
    python tools/test.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py  checkpoints/SOME_CHECKPOINT.pth  --eval AR@AN --out results.json
    ```

## 如何测试

用户可以使用以下指令进行模型测试。

1. TEM

    ```shell
    # 注：该命令无法进行指标验证
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

例如：

1. 在 ActivityNet 数据集上测试 TEM 模型。

    ```shell
    python tools/test.py configs/localization/bsn/bsn_tem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth
    ```

2. 在 ActivityNet 数据集上测试 PGM 模型。

    ```shell
    python tools/misc/bsn_proposal_generation.py configs/localization/bsn/bsn_pgm_400x100_activitynet_feature.py --mode test
    ```

3. 测试 PEM 模型，并计算 'AR@AN' 指标，输出结果文件。

    ```shell
    python tools/test.py configs/localization/bsn/bsn_pem_400x100_1x16_20e_activitynet_feature.py checkpoints/SOME_CHECKPOINT.pth --eval AR@AN --out results.json
    ```

注：

1. (可选项) 用户可以使用以下指令生成格式化的时序动作候选文件，该文件可被送入动作识别器中（目前只支持 SSN 和 P-GCN，不包括 TSN, I3D 等），以获得时序动作候选的分类结果。

    ```shell
    python tools/data/activitynet/convert_proposal_format.py
    ```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
