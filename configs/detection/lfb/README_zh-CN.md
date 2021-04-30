# LFB

## 简介

<!-- [ALGORITHM] -->

```BibTeX
@inproceedings{wu2019long,
  title={Long-term feature banks for detailed video understanding},
  author={Wu, Chao-Yuan and Feichtenhofer, Christoph and Fan, Haoqi and He, Kaiming and Krahenbuhl, Philipp and Girshick, Ross},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={284--293},
  year={2019}
}
```

## 模型库

### AVA2.1

| 配置文件 | 模态 |  预训练  | 主干网络  | 输入 | GPU 数量 | 分辨率 | 平均精度 | log | json | ckpt |
| :----------------------------------------------------------: | :------: | :----------: | :-------: | :---: | :--: | :------------: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16 | 8 | 短边 256 | 24.11 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210224_125052.log) | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210224_125052.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210224-2ae136d9.pth) |
| [lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16 | 8 | 短边 256 | 20.17 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log) | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_avg_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210301-19c330b7.pth) |
| [lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py](/configs/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py) |   RGB    | Kinetics-400 | [slowonly_r50_4x16x1](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) | 4x16 | 8 | 短边 256 | 22.15 | [log](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log) | [json](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/20210301_124812.log.json) | [ckpt](https://download.openmmlab.com/mmaction/detection/lfb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb/lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb_20210301-37efcd15.pth) |

- 注:

1. 这里的 **GPU 数量** 指的是得到模型权重文件对应的 GPU 个数。默认地，MMAction2 所提供的配置文件对应使用 8 块 GPU 进行训练的情况。
   依据 [线性缩放规则](https://arxiv.org/abs/1706.02677)，当用户使用不同数量的 GPU 或者每块 GPU 处理不同视频个数时，需要根据批大小等比例地调节学习率。
   如，lr=0.01 对应 4 GPUs x 2 video/gpu，以及 lr=0.08 对应 16 GPUs x 4 video/gpu。
2. 本 LFB 模型暂没有使用原论文中的 `I3D-R50-NL` 作为主干网络，而是用 `slowonly_r50_4x16x1` 替代，但取得了同样的提升效果：（本模型：20.1 -> 24.11 而原论文模型：22.1 -> 25.8）。
3. 因为测试时，长时特征是被随机采样的，所以测试精度可能有一些偏差。
4. 在训练或测试 LFB 之前，用户需要使用配置文件特征库 [lfb_slowonly_r50_ava_infer.py](/configs/detection/lfb/lfb_slowonly_r50_ava_infer.py) 来推导长时特征库。有关推导长时特征库的更多细节，请参照[训练部分](#训练)。
5. 用户也可以直接从 [AVA_train_val_float32_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float32_lfb.rar) 或者 [AVA_train_val_float16_lfb](https://download.openmmlab.com/mmaction/detection/lfb/AVA_train_val_float16_lfb.rar) 下载 float32 或 float16 的长时特征库，并把它们放在 `lfb_prefix_path` 上。

## 训练

### a. 为训练 LFB 推导长时特征库

在训练或测试 LFB 之前，用户首先需要推导长时特征库。

具体来说，使用配置文件 [lfb_slowonly_r50_ava_infer](/configs/detection/lfb/lfb_slowonly_r50_ava_infer.py)，在训练集、验证集、测试集上都运行一次模型测试。

配置文件的默认设置是推导训练集的长时特征库，用户需要将 `dataset_mode` 设置成 `'val'` 来推导验证集的长时特征库，在推导过程中。共享头 [LFBInferHead](/mmaction/models/heads/lfb_infer_head.py) 会生成长时特征库。

AVA 训练集和验证集的 float32 精度的长时特征库文件大约占 3.3 GB。如果以半精度来存储长时特征，文件大约占 1.65 GB。

用户可以使用以下命令来推导 AVA 训练集和验证集的长时特征库，而特征库会被存储为 `lfb_prefix_path/lfb_train.pkl` 和 `lfb_prefix_path/lfb_val.pkl`。

```shell
# 在 lfb_slowonly_r50_ava_infer.py 中 设置 `dataset_mode = 'train'`
python tools/test.py configs/detection/lfb/lfb_slowonly_r50_ava_infer.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth --eval mAP

# 在 lfb_slowonly_r50_ava_infer.py 中 设置 `dataset_mode = 'val'`
python tools/test.py configs/detection/lfb/lfb_slowonly_r50_ava_infer.py \
    checkpoints/YOUR_BASELINE_CHECKPOINT.pth --eval mAP
```

MMAction2 使用来自配置文件 [slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb](/configs/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb.py) 的模型权重文件 [slowonly_r50_4x16x1 checkpoint](https://download.openmmlab.com/mmaction/detection/ava/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb/slowonly_kinetics_pretrained_r50_4x16x1_20e_ava_rgb_20201217-40061d5f.pth)作为推导长时特征库的 LFB 模型的主干网络的预训练模型。

### b. 训练 LFB

用户可以使用以下指令进行模型训练。

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

例如：使用半精度的长时特征库在 AVA 数据集上训练 LFB 模型。

```shell
python tools/train.py configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py \
  --validate --seed 0 --deterministic
```

更多训练细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#训练配置) 中的 **训练配置** 部分。

## 测试

### a. 为测试 LFB 推导长时特征库

在训练或测试 LFB 之前，用户首先需要推导长时特征库。如果用户之前已经生成了特征库文件，可以跳过这一步。

这一步做法与[训练部分](#Train)中的 **为训练 LFB 推导长时特征库** 相同。

### b. 测试 LFB

用户可以使用以下指令进行模型测试。

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

例如：使用半精度的长时特征库在 AVA 数据集上测试 LFB 模型，并将结果导出为一个 json 文件。

```shell
python tools/test.py configs/detection/lfb/lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval mAP --out results.csv
```

更多测试细节，可参考 [基础教程](/docs_zh_CN/getting_started.md#测试某个数据集) 中的 **测试某个数据集** 部分。
