# AGCN

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{shi2019two,
  title={Two-stream adaptive graph convolutional networks for skeleton-based action recognition},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12026--12035},
  year={2019}
}
```

## 模型库

### NTU60_XSub

| 配置文件                                                                                            | 数据格式 | GPU 数量 | 主干网络 | top1 准确率 |                                                                       ckpt                                                                        |                                                                   log                                                                   |                                                                   json                                                                    |
| :-------------------------------------------------------------------------------------------------- | :------: | :------: | :------: | :---------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| [2sagcn_80e_ntu60_xsub_keypoint_3d](/configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py) |  joint   |    1     |   AGCN   |    86.06    | [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d-3bed61ba.pth) | [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.log) | [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d/2sagcn_80e_ntu60_xsub_keypoint_3d.json) |
| [2sagcn_80e_ntu60_xsub_bone_3d](/configs/skeleton/ss-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py)         |   bone   |    2     |   AGCN   |    86.89    |     [ckpt](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d-278b8815.pth)     |     [log](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.log)     |     [json](https://download.openmmlab.com/mmaction/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d/2sagcn_80e_ntu60_xsub_bone_3d.json)     |

## 如何训练

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 AGCN 模型在 NTU60 数据集的骨骼数据上的训练。

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_keypoint_3d \
    --validate --seed 0 --deterministic
```

例如：以一个确定性的训练方式，辅以定期的验证过程进行 AGCN 模型在 NTU60 数据集的关节数据上的训练。

```shell
python tools/train.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    --work-dir work_dirs/2sagcn_80e_ntu60_xsub_bone_3d \
    --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 如何测试

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：在 NTU60 数据集的骨骼数据上测试 AGCN 模型，并将结果导出为一个 pickle 文件。

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_keypoint_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out joint_result.pkl
```

例如：在 NTU60 数据集的关节数据上测试 AGCN 模型，并将结果导出为一个 pickle 文件。

```shell
python tools/test.py configs/skeleton/2s-agcn/2sagcn_80e_ntu60_xsub_bone_3d.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out bone_result.pkl
```

更多测试细节，可参考 [基础教程](/docs/zh_cn/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
