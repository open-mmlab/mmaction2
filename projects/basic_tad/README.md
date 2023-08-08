# Example Project

This is an example README for community `projects/`. You can write your README in your own project. Here are
some recommended parts of a README for others to understand and use your project, you can copy or modify them
according to your project.

## Usage

### Setup Environment

Please refer to [Get Started](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) to install MMAction2.

At first, add the current folder to `PYTHONPATH`, so that Python can find your code. Run command in the current directory to add it.

> Please run it every time after you opened a new shell.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```

### Data Preparation

Prepare the Kinetics400 dataset according to the [instruction](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md).

### Training commands

**To train with single GPU:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py
```

**To train with multiple GPUs:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --launcher pytorch --gpus 8
```

**To train with multiple GPUs by slurm:**

```bash
mim train mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

### Testing commands

**To test with single GPU:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT
```

**To test with multiple GPUs:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher pytorch --gpus 8
```

**To test with multiple GPUs by slurm:**

```bash
mim test mmaction configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py --checkpoint $CHECKPOINT --launcher slurm \
    --gpus 8 --gpus-per-node 8 --partition $PARTITION
```

## Results

| frame sampling strategy | resolution | gpus | backbone | pretrain | top1 acc | top5 acc |  testing protocol  |                    config                     |                                   ckpt |                            log |
| :---------------------: | :--------: | :--: | :------: | :------: | :------: | :------: | :----------------: | :-------------------------------------------: | -------------------------------------: | -----------------------------: |
|          1x1x3          |  224x224   |  8   | ResNet50 | ImageNet |  72.83   |  90.65   | 25 clips x 10 crop | [config](./configs/examplenet_r50-in1k-pre_8xb32-1x1x3-100e_kinetics400-rgb.py) | [ckpt](https://example/checkpoint/url) | [log](https://example/log/url) |

## Citation

<!-- Replace to the citation of the paper your project refers to. -->

```bibtex
@misc{2020mmaction2,
  title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
  author={MMAction2 Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
  year={2020}
}
```

## Checklist

Here is a checklist of this project's progress, and you can ignore this part if you don't plan to contribute to MMAction2 projects.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

    <!-- The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmaction.registry.MODELS` and configurable via a config file. -->

  - [ ] Basic docstrings & proper citation

    <!-- Each major class should contains a docstring, describing its functionality and arguments. If your code is copied or modified from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd) -->

  - [ ] Converted checkpoint and results (Only for reproduction)

    <!-- If you are reproducing the result from a paper, make sure the model in the project can match that results. Also please provide checkpoint links or a checkpoint conversion script for others to get the pre-trained model. -->

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training results

    <!-- If you are reproducing the result from a paper, train your model from scratch and verified that the final result can match the original result. Usually, ±0.1% is acceptable for the action recognition task on Kinetics400. -->

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Unit tests

    <!-- Unit tests for the major module are required. [Example](https://github.com/open-mmlab/mmaction2/blob/main/tests/models/backbones/test_resnet.py) -->

  - [ ] Code style

    <!-- Refactor your code according to reviewer's comment. -->

  - [ ] `metafile.yml` and `README.md`

    <!-- It will used for MMAction2 to acquire your models. [Example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/metafile.yml). In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/README.md) -->
