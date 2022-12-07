# STGCN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
```

## 模型库

### NTU60_XSub

| 配置文件                                                                                        | 骨骼点 | GPU 数量 | 主干网络 | Top-1 准确率 |                                                                     ckpt                                                                      |                                                                 log                                                                 |                                                                 json                                                                  |
| :---------------------------------------------------------------------------------------------- | :----: | :------: | :------: | :----------: | :-------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------: |
| [stgcn_80e_ntu60_xsub_keypoint](/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py)       |   2d   |    2     |  STGCN   |    86.91     |    [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth)    |    [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.log)    |    [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint/stgcn_80e_ntu60_xsub_keypoint.json)    |
| [stgcn_80e_ntu60_xsub_keypoint_3d](/configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d.py) |   3d   |    1     |  STGCN   |    84.61     | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d-13e7ccf0.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d/stgcn_80e_ntu60_xsub_keypoint_3d.json) |

### BABEL

| 配置文件                                                                    | GPU 数量 | 主干网络 | Top-1 准确率 | 类平均 Top-1 准确率 | Top-1 准确率 <br>（官方，使用 AGCN） | 类平均 Top-1 准确率<br>（官方，使用 AGCN） |                                                           ckpt                                                            |                                                    log                                                     |
| --------------------------------------------------------------------------- | :------: | :------: | :----------: | :-----------------: | :----------------------------------: | :----------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
| [stgcn_80e_babel60](/configs/skeleton/stgcn/stgcn_80e_babel60.py)           |    8     |  ST-GCN  |  **42.39**   |      **28.28**      |                41.14                 |                   24.46                    |      [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60-3d206418.pth)      |   [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60.log)    |
| [stgcn_80e_babel60_wfl](/configs/skeleton/stgcn/stgcn_80e_babel60_wfl.py)   |    8     |  ST-GCN  |  **40.31**   |        29.79        |                33.41                 |                 **30.42**                  |  [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60_wfl/stgcn_80e_babel60_wfl-1a9102d7.pth)  | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel60_wfl.log)  |
| [stgcn_80e_babel120](/configs/skeleton/stgcn/stgcn_80e_babel120.py)         |    8     |  ST-GCN  |  **38.95**   |      **20.58**      |                38.41                 |                   17.56                    |     [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120/stgcn_80e_babel120-e41eb6d7.pth)     |   [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120.log)   |
| [stgcn_80e_babel120_wfl](/configs/skeleton/stgcn/stgcn_80e_babel120_wfl.py) |    8     |  ST-GCN  |  **33.00**   |        24.33        |                27.91                 |                **26.17**\*                 | [ckpt](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel120_wfl/stgcn_80e_babel120_wfl-3f2c100d.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/stgcn/stgcn_80e_babel60/stgcn_80e_babel120_wfl.log) |

\* 注：此数字引自原 [论文](https://arxiv.org/pdf/2106.09696.pdf)， 实际公开的 [模型权重](https://github.com/abhinanda-punnakkal/BABEL/tree/main/action_recognition) 精度略低一些。

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 STGCN 模型在 NTU60 数据集上的训练

```shell
python tools/train.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    --work-dir work_dirs/stgcn_80e_ntu60_xsub_keypoint \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 NTU60 数据集上测试 STGCN 模型，并将结果导出为一个 pickle 文件。

```shell
python tools/test.py configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.pkl
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
