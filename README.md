<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_logo.png" width="500"/>
</div>

## Introduction

English | [简体中文](/README_zh-CN.md)

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

MMAction2 is an open-source toolbox for video understanding based on PyTorch.
It is a part of the [OpenMMLab](http://openmmlab.org/) project.

The master branch works with **PyTorch 1.3+**.

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_overview.gif" width="450px"/><br>
    Action Recognition Results on Kinetics-400
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/><br>
    Spatio-Temporal Action Detection Results on AVA-2.1
</div>
<div align="center">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="500px"/><br>
    Skeleton-base Action Recognition Results on NTU-RGB+D-120
</div>

### Major Features

- **Modular design**

  We decompose the video understanding framework into different components and one can easily construct a customized
  video understanding framework by combining different modules.

- **Support for various datasets**

  The toolbox directly supports multiple datasets, UCF101, Kinetics-[400/600/700], Something-Something V1&V2, Moments in Time, Multi-Moments in Time, THUMOS14, etc.

- **Support for multiple video understanding frameworks**

  MMAction2 implements popular frameworks for video understanding:

  - For action recognition, various algorithms are implemented, including TSN, TSM, TIN, R(2+1)D, I3D, SlowOnly, SlowFast, CSN, Non-local, etc.

  - For temporal action localization, we implement BSN, BMN, SSN.

  - For spatial temporal detection, we implement SlowOnly, SlowFast.

- **Well tested and documented**

  We provide detailed documentation and API reference, as well as unittests.

## Changelog

v0.18.0 was released in 02/09/2021. Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark

| Model  |input| io backend | batch size x gpus | MMAction2 (s/iter) | MMAction (s/iter) | Temporal-Shift-Module (s/iter) | PySlowFast (s/iter) |
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  | :--------------------: | :----------------------------: | :-----------------: |
| [TSN](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py)| 256p rawframes |Memcached| 32x8|**[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_rawframes_memcahed_32x8.zip)** | [0.38](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/tsn_256p_rawframes_memcached_32x8.zip)| [0.42](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsn_256p_rawframes_memcached_32x8.zip)| x |
| [TSN](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py)| 256p dense-encoded video |Disk| 32x8|**[0.61](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_fast_videos_disk_32x8.zip)**| x | x | TODO |
|[I3D heavy](/configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py)|256p videos|Disk |8x8| **[0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_heavy_256p_videos_disk_8x8.zip)** | x | x | [0.44](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_i3d_r50_8x8_video.log) |
| [I3D](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)|256p rawframes|Memcached|8x8| **[0.43](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_256p_rawframes_memcahed_8x8.zip)** | [0.56](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/i3d_256p_rawframes_memcached_8x8.zip) | x | x |
| [TSM](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |256p rawframes|Memcached| 8x8|**[0.31](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsm_256p_rawframes_memcahed_8x8.zip)** | x | [0.41](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsm_256p_rawframes_memcached_8x8.zip) | x |
| [Slowonly](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowonly_256p_videos_disk_8x8.zip)** | TODO | x | [0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowonly_r50_4x16_video.log) |
| [Slowfast](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.69](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowfast_256p_videos_disk_8x8.zip)** | x | x | [1.04](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowfast_r50_4x16_video.log) |
| [R(2+1)D](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py)|256p videos |Disk| 8x8|**[0.45](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/r2plus1d_256p_videos_disk_8x8.zip)** | x | x | x |

Details can be found in [benchmark](docs/benchmark.md).

## ModelZoo

Supported methods for Action Recognition:

<details open>
<summary>(click to collapse)</summary>

- ✅ [TSN](configs/recognition/tsn/README.md) (ECCV'2016)
- ✅ [TSM](configs/recognition/tsm/README.md) (ICCV'2019)
- ✅ [TSM Non-Local](configs/recognition/tsm/README.md) (ICCV'2019)
- ✅ [R(2+1)D](configs/recognition/r2plus1d/README.md) (CVPR'2018)
- ✅ [I3D](configs/recognition/i3d/README.md) (CVPR'2017)
- ✅ [I3D Non-Local](configs/recognition/i3d/README.md) (CVPR'2018)
- ✅ [SlowOnly](configs/recognition/slowonly/README.md) (ICCV'2019)
- ✅ [SlowFast](configs/recognition/slowfast/README.md) (ICCV'2019)
- ✅ [CSN](configs/recognition/csn/README.md) (ICCV'2019)
- ✅ [TIN](configs/recognition/tin/README.md) (AAAI'2020)
- ✅ [TPN](configs/recognition/tpn/README.md) (CVPR'2020)
- ✅ [C3D](configs/recognition/c3d/README.md) (CVPR'2014)
- ✅ [X3D](configs/recognition/x3d/README.md) (CVPR'2020)
- ✅ [OmniSource](configs/recognition/omnisource/README.md) (ECCV'2020)
- ✅ [MultiModality: Audio](configs/recognition_audio/resnet/README.md) (ArXiv'2020)
- ✅ [TANet](configs/recognition/tanet/README.md) (ArXiv'2020)
- ✅ [TRN](configs/recognition/trn/README.md) (CVPR'2015)
- ✅ [Timesformer](configs/recognition/timesformer/README.md) (ICML'2021)

</details>

Supported methods for Temporal Action Detection:

<details open>
<summary>(click to collapse)</summary>

- ✅ [BSN](configs/localization/bsn/README.md) (ECCV'2018)
- ✅ [BMN](configs/localization/bmn/README.md) (ICCV'2019)
- ✅ [SSN](configs/localization/ssn/README.md) (ICCV'2017)

</details>

Supported methods for Spatial Temporal Action Detection:

<details open>
<summary>(click to collapse)</summary>

- ✅ [ACRN](configs/detection/acrn/README.md) (ECCV'2018)
- ✅ [SlowOnly+Fast R-CNN](configs/detection/ava/README.md) (ICCV'2019)
- ✅ [SlowFast+Fast R-CNN](configs/detection/ava/README.md) (ICCV'2019)
- ✅ [Long-Term Feature Bank](configs/detection/lfb/README.md) (CVPR'2019)

</details>

Supported methods for Skeleton-based Action Recognition:

<details open>
<summary>(click to collapse)</summary>

- ✅ [PoseC3D](configs/skeleton/posec3d/README.md) (ArXiv'2021)
- ✅ [STGCN](configs/skeleton/stgcn/README.md) (AAAI'2018)

</details>

Results and models are available in the *README.md* of each method's config directory.
A summary can be found in the [**model zoo**](https://mmaction2.readthedocs.io/en/latest/recognition_models.html) page.

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks.
If you have any feature requests, please feel free to leave a comment in [Issues](https://github.com/open-mmlab/mmaction2/issues/19).

## Dataset

Supported [datasets](https://mmaction2.readthedocs.io/en/latest/supported_datasets.html):

Supported datasets for Action Recognition:

<details open>
<summary>(click to collapse)</summary>

- ✅ [UCF101](/tools/data/ucf101/README.md) \[ [Homepage](https://www.crcv.ucf.edu/research/data-sets/ucf101/) \] (CRCV-IR-12-01)
- ✅ [HMDB51](/tools/data/hmdb51/README.md) \[ [Homepage](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) \] (ICCV'2011)
- ✅ [Kinetics-[400/600/700]](/tools/data/kinetics/README.md) \[ [Homepage](https://deepmind.com/research/open-source/kinetics) \] (CVPR'2017)
- ✅ [Something-Something V1](/tools/data/sthv1/README.md) \[ [Homepage](https://20bn.com/datasets/something-something/v1) \] (ICCV'2017)
- ✅ [Something-Something V2](/tools/data/sthv2/README.md) \[ [Homepage](https://20bn.com/datasets/something-something) \] (ICCV'2017)
- ✅ [Moments in Time](/tools/data/mit/README.md) \[ [Homepage](http://moments.csail.mit.edu/) \] (TPAMI'2019)
- ✅ [Multi-Moments in Time](/tools/data/mmit/README.md) \[ [Homepage](http://moments.csail.mit.edu/challenge_iccv_2019.html) \] (ArXiv'2019)
- ✅ [HVU](/tools/data/hvu/README.md) \[ [Homepage](https://github.com/holistic-video-understanding/HVU-Dataset) \] (ECCV'2020)
- ✅ [Jester](/tools/data/jester/README.md) \[ [Homepage](https://20bn.com/datasets/jester/v1) \] (ICCV'2019)
- ✅ [GYM](/tools/data/gym/README.md) \[ [Homepage](https://sdolivia.github.io/FineGym/) \] (CVPR'2020)
- ✅ [ActivityNet](/tools/data/activitynet/README.md) \[ [Homepage](http://activity-net.org/) \] (CVPR'2015)
- ✅ [Diving48](/tools/data/diving48/README.md) \[ [Homepage](http://www.svcl.ucsd.edu/projects/resound/dataset.html) \] (ECCV'2018)
- ✅ [OmniSource](/tools/data/omnisource/README.md) \[ [Homepage](https://kennymckormick.github.io/omnisource/) \] (ECCV'2020)

</details>

Supported datasets for Temporal Action Detection

<details open>
<summary>(click to collapse)</summary>

- ✅ [ActivityNet](/tools/data/activitynet/README.md) \[ [Homepage](http://activity-net.org/) \] (CVPR'2015)
- ✅ [THUMOS14](/tools/data/thumos14/README.md) \[ [Homepage](https://www.crcv.ucf.edu/THUMOS14/download.html) \] (THUMOS Challenge 2014)

</details>

Supported datasets for Spatial Temporal Action Detection

<details open>
<summary>(click to collapse)</summary>

- ✅ [AVA](/tools/data/ava/README.md) \[ [Homepage](https://research.google.com/ava/index.html) \] (CVPR'2018)
- 🔲 [UCF101-24](/tools/data/ucf101_24/README.md) \[ [Homepage](http://www.thumos.info/download.html) \] (CRCV-IR-12-01)
- 🔲 [JHMDB](/tools/data/jhmdb/README.md) \[ [Homepage](http://jhmdb.is.tue.mpg.de/) \] (ICCV'2013)

</details>

Supported datasets for Skeleton-based Action Detection

<details open>
<summary>(click to collapse)</summary>

- ✅ [PoseC3D-FineGYM](/tools/data/skeleton/README.md) \[ [Homepage](https://kennymckormick.github.io/posec3d/) \] (arXiv'2021)

</details>

Datasets marked with 🔲 are not fully supported yet, but related dataset preparation steps are provided.

## Installation

Please refer to [install.md](docs/install.md) for installation.

## Data Preparation

Please refer to [data_preparation.md](docs/data_preparation.md) for a general knowledge of data preparation.
The supported datasets are listed in [supported_datasets.md](docs/supported_datasets.md)

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMAction2.
There are also tutorials:

- [learn about configs](docs/tutorials/1_config.md)
- [finetuning models](docs/tutorials/2_finetune.md)
- [adding new dataset](docs/tutorials/3_new_dataset.md)
- [designing data pipeline](docs/tutorials/4_data_pipeline.md)
- [adding new modules](docs/tutorials/5_new_modules.md)
- [exporting model to onnx](docs/tutorials/6_export_model.md)
- [customizing runtime settings](docs/tutorials/7_customize_runtime.md)

A Colab tutorial is also provided. You may preview the notebook [here](demo/mmaction2_tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb) on Colab.

## FAQ

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

## Contributing

We appreciate all contributions to improve MMAction2. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) in MMCV for more details about the contributing guideline.

## Acknowledgement

MMAction2 is an open source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new models.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation video understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
