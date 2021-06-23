# Preparing MultiSports

## Introduction

```BibTex
@article{li2021multisports,
  title={MultiSports: A Multi-Person Video Dataset of Spatio-Temporally Localized Sports Actions},
  author={Li, Yixuan and Chen, Lei and He, Runyu and Wang, Zhenzhi and Wu, Gangshan and Wang, Limin},
  journal={arXiv preprint arXiv:2105.07404},
  year={2021}
}
```

For basic dataset information, please refer to the [official website](https://deeperaction.github.io/multisports/)
and the [paper](https://arxiv.org/abs/2105.07404). Or you can seek more information in related [challenge](https://competitions.codalab.org/competitions/32066),
held from June 1, 2021 to Sept 10, 2021.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/multisports/`.

## Download and uncompress

xxxx

## Extract Frames

xxxx

## Check Directory Structure

xxxx

In the context of the whole project (for MultiSports only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── multisports
```

**Note**: The `multisports_GT.pkl` exists as a cache, it contains 6 items as follows:

1. `labels` (list):
2. `gttubes` (dict):
3. `nframes` (dict)
4. `train_videos` (list)
5. `test_videos` (list)
6. `resolution` (dict)
