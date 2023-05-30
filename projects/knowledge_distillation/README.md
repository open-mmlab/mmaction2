# Knowledge Distillation Based on MMRazor

Knowledge Distillation is a classic model compression method. The core idea is to "imitate" a teacher model (or multi-model ensemble) with better performance and more complex structure by guiding a lightweight student model, improving the performance of the student model without changing its structure. [MMRazor](https://github.com/open-mmlab/mmrazor) is a model compression toolkit for model slimming and AutoML, which supports several KD algorithms. In this project, we take TSM-MobileNetV2 as an example to show how to use MMRazor to perform knowledge distillation on action recognition models. You could refer to more [MMRazor](https://github.com/open-mmlab/mmrazor) for more model compression algorithms.

## Description

This is an implementation of MMRazor Knowledge Distillation Application, we provide action recognition configs and models for MMRazor.

## Usage

### Prerequisites

- [MMRazor v1.0.0](https://github.com/open-mmlab/mmrazor/tree/v1.0.0) or higher

There are two install modes:

Option (a). Install as a Python package

```shell
mim install "mmrazor>=1.0.0"
```

Option (b). Install from source

```shell
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .
```

### Setup Environment

Please refer to [Get Started](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

Create a symbolic link from `$MMACTION2/data` to `./data` in the current directory, so that Python can locate your data. Run the following command in the current directory to create the symbolic link.

```shell
ln -s ../../data ./data
```

### Training commands

**To train with single GPU:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py
```

**To train with multiple GPUs:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmrazor configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

Please convert the knowledge distillation checkpoint to student-only checkpoint with following commands, you will get a checkpoint with a '\_student.pth' suffix under the same directory as the original checkpoint. Then take the student-only checkpoint for testing.

```bash
mim run mmrazor convert_kd_ckpt_to_student $CHECKPOINT
```

**To test with single GPU:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results and models

| Location |   Dataset    |    Teacher     |      Student      |     Acc     | Acc(T) | Acc(S) |        Config         | Download                                                                      |
| :------: | :----------: | :------------: | :---------------: | :---------: | :----: | :----: | :-------------------: | :---------------------------------------------------------------------------- |
|  logits  | Kinetics-400 | [TSM-ResNet50] | [TSM-MobileNetV2] | 69.60(+0.9) | 73.22  | 68.71  | [config][distill_tsm] | [teacher][tsm_r50_pth] \| [model][distill_pth_tsm] \| [log][distill_log_tsm]  |
|  logits  | Kinetics-400 |   [TSN-Swin]   |  [TSN-ResNet50]   | 75.54(+1.4) | 79.22  | 74.12  | [config][distill_tsn] | [teacher][tsn_swin_pth] \| [model][distill_pth_tsn] \| [log][distill_log_tsn] |

## Citation

```latex
@article{huang2022knowledge,
  title={Knowledge Distillation from A Stronger Teacher},
  author={Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2205.10536},
  year={2022}
}
```

[distill_log_tsm]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.log
[distill_log_tsn]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsn-swin_tsn-r50_1x1x8_k400/kd_logits_tsn-swin_tsn-r50_1x1x8_k400.log
[distill_pth_tsm]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400_20230517-c3e8aa0d.pth
[distill_pth_tsn]: https://download.openmmlab.com/mmaction/v1.0/projects/knowledge_distillation/kd_logits_tsn-swin_tsn-r50_1x1x8_k400/kd_logits_tsn-swin_tsn-r50_1x1x8_k400_student_20230530-f938d404.pth
[distill_tsm]: configs/kd_logits_tsm-res50_tsm-mobilenetv2_8xb16_k400.py
[distill_tsn]: configs/kd_logits_tsn-swin_tsn-r50_8xb16_k400.py
[tsm-mobilenetv2]: ../../configs/recognition/tsm/tsm_imagenet-pretrained-mobilenetv2_8xb16-1x1x8-100e_kinetics400-rgb.py
[tsm-resnet50]: ../../configs/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-50e_kinetics400-rgb.py
[tsm_r50_pth]: https://download.openmmlab.com/mmaction/v1.0/recognition/tsm/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb/tsm_imagenet-pretrained-r50_8xb16-1x1x8-100e_kinetics400-rgb_20220831-a6db1e5d.pth
[tsn-resnet50]: ../../configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py
[tsn-swin]: ../../configs/recognition/tsn/custom_backbones/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb.py
[tsn_swin_pth]: https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb/tsn_imagenet-pretrained-swin-transformer_32xb8-1x1x8-50e_kinetics400-rgb_20230530-428f0064.pth
