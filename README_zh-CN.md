<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_logo.png" width="500"/>
</div>

## 简介

[English](/README.md) | 简体中文

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/zh_CN/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

MMAction2 是一款基于 PyTorch 的视频理解开源工具箱，是 [OpenMMLab](http://openmmlab.org/) 项目的成员之一

主分支代码目前支持 **PyTorch 1.3 以上**的版本

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/mmaction2_overview.gif" width="450px"/><br>
    Kinetics-400 上的动作识别
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/spatio-temporal-det.gif" width="800px"/><br>
    AVA-2.1 上的时空动作检测
</div>
<div align="center">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="500px"/><br>
    NTURGB+D-120 上的基于人体姿态的动作识别
</div>

## 主要特性

- **模块化设计**
  MMAction2 将统一的视频理解框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的视频理解模型

- **支持多样的数据集**
  MMAction2 支持多种数据集的准备和构建，包括 UCF101, Kinetics-[400/600/700], Something-Something V1&V2, Moments in Time, Multi-Moments in Time, THUMOS14 等

- **支持多种视频理解任务**
  MMAction2 支持多种主流的视频理解任务

  - 动作识别：TSN, TSM, TIN, R(2+1)D, I3D, SlowOnly, SlowFast, CSN, Non-local 等
  - 时序动作检测：BSN, BMN, SSN
  - 时空动作检测：SlowOnly + Fast-RCNN, SlowFast + Fast-RCNN

  具体可参考 [模型库](#模型库)

- **详尽的单元测试和文档**
  MMAction2 提供了详尽的说明文档，API 接口说明，全面的单元测试，以供社区参考

## 相关工作

目前有许多研究工作或工程项目基于 MMAction2 搭建，例如：

- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [[论文]](https://arxiv.org/abs/2107.10161)[[代码]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [[论文]](https://arxiv.org/abs/2103.17263)[[代码]](https://github.com/xvjiarui/VFS)
- Video Swin Transformer. [[论文]](https://arxiv.org/abs/2106.13230)[[代码]](https://github.com/SwinTransformer/Video-Swin-Transformer)

更多详情可见 [projects.md](docs/projects.md)

## 更新记录

v0.19.0 版本已于 2021 年 10 月 7 日发布，可通过查阅 [更新日志](/docs/changelog.md) 了解更多细节以及发布历史

## 基准测试

为了验证 MMAction2 框架的高精度和高效率，开发成员将其与当前其他主流框架进行速度对比。更多详情可见 [基准测试](/docs_zh_CN/benchmark.md)

## 模型库

支持的行为识别方法：

<details open>
<summary>(点击收起)</summary>

- ✅ [TSN](/configs/recognition/tsn/README_zh-CN.md) (ECCV'2016)
- ✅ [TSM](/configs/recognition/tsm/README_zh-CN.md) (ICCV'2019)
- ✅ [TSM Non-Local](/configs/recognition/tsm/README_zh-CN.md) (ICCV'2019)
- ✅ [R(2+1)D](/configs/recognition/r2plus1d/README_zh-CN.md) (CVPR'2018)
- ✅ [I3D](/configs/recognition/i3d/README_zh-CN.md) (CVPR'2017)
- ✅ [I3D Non-Local](/configs/recognition/i3d/README_zh-CN.md) (CVPR'2018)
- ✅ [SlowOnly](/configs/recognition/slowonly/README_zh-CN.md) (ICCV'2019)
- ✅ [SlowFast](/configs/recognition/slowfast/README_zh-CN.md) (ICCV'2019)
- ✅ [CSN](/configs/recognition/csn/README_zh-CN.md) (ICCV'2019)
- ✅ [TIN](/configs/recognition/tin/README_zh-CN.md) (AAAI'2020)
- ✅ [TPN](/configs/recognition/tpn/README_zh-CN.md) (CVPR'2020)
- ✅ [C3D](/configs/recognition/c3d/README_zh-CN.md) (CVPR'2014)
- ✅ [X3D](/configs/recognition/x3d/README_zh-CN.md) (CVPR'2020)
- ✅ [OmniSource](/configs/recognition/omnisource/README_zh-CN.md) (ECCV'2020)
- ✅ [MultiModality: Audio](/configs/recognition_audio/resnet/README_zh-CN.md) (ArXiv'2020)
- ✅ [TANet](/configs/recognition/tanet/README_zh-CN.md) (ArXiv'2020)
- ✅ [TRN](/configs/recognition/trn/README_zh-CN.md) (ECCV'2018)
- ✅ [Timesformer](configs/recognition/timesformer/README_zh-CN.md) (ICML'2021)

</details>

支持的时序动作检测方法：

<details open>
<summary>(点击收起)</summary>

- ✅ [BSN](/configs/localization/bsn/README_zh-CN.md) (ECCV'2018)
- ✅ [BMN](/configs/localization/bmn/README_zh-CN.md) (ICCV'2019)
- ✅ [SSN](/configs/localization/ssn/README_zh-CN.md) (ICCV'2017)

</details>

支持的时空检测方法：

<details open>
<summary>(点击收起)</summary>

- ✅ [ACRN](configs/detection/acrn/README_zh-CN.md) (ECCV'2018)
- ✅ [SlowOnly+Fast R-CNN](/configs/detection/ava/README_zh-CN.md) (ICCV'2019)
- ✅ [SlowFast+Fast R-CNN](/configs/detection/ava/README_zh-CN.md) (ICCV'2019)
- ✅ [Long-Term Feature Bank](/configs/detection/lfb/README_zh-CN.md) (CVPR'2019)

</details>

支持的基于骨骼点的动作识别方法:

<details open>
<summary>(点击收起)</summary>

- ✅ [PoseC3D](configs/skeleton/posec3d/README.md) (ArXiv'2021)
- ✅ [STGCN](configs/skeleton/stgcn/README.md) (AAAI'2018)

</details>

各个模型的结果和设置都可以在对应的 config 目录下的 *README_zh-CN.md* 中查看。整体的概况也可也在 [**模型库**](https://mmaction2.readthedocs.io/zh_CN/latest/recognition_models.html) 页面中查看

我们将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMAction2 有任何功能需求，请随时在 [问题](https://github.com/open-mmlab/mmaction2/issues/19) 中留言。

## 数据集

支持的 [数据集](https://mmaction2.readthedocs.io/zh_CN/latest/supported_datasets.html)：

支持的动作识别数据集：

<details open>
<summary>(点击收起)</summary>

- ✅ [UCF101](/tools/data/ucf101/README_zh-CN.md) \[ [主页](https://www.crcv.ucf.edu/research/data-sets/ucf101/) \] (CRCV-IR-12-01)
- ✅ [HMDB51](/tools/data/hmdb51/README_zh-CN.md) \[ [主页](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) \] (ICCV'2011)
- ✅ [Kinetics-[400/600/700]](/tools/data/kinetics/README_zh-CN.md) \[ [主页](https://deepmind.com/research/open-source/kinetics) \] (CVPR'2017)
- ✅ [Something-Something V1](/tools/data/sthv1/README_zh-CN.md) \[ [主页](https://20bn.com/datasets/something-something/v1) \] (ICCV'2017)
- ✅ [Something-Something V2](/tools/data/sthv2/README_zh-CN.md) \[ [主页](https://20bn.com/datasets/something-something) \] (ICCV'2017)
- ✅ [Moments in Time](/tools/data/mit/README_zh-CN.md) \[ [主页](http://moments.csail.mit.edu/) \] (TPAMI'2019)
- ✅ [Multi-Moments in Time](/tools/data/mmit/README_zh-CN.md) \[ [主页](http://moments.csail.mit.edu/challenge_iccv_2019.html) \] (ArXiv'2019)
- ✅ [HVU](/tools/data/hvu/README_zh-CN.md) \[ [主页](https://github.com/holistic-video-understanding/HVU-Dataset) \] (ECCV'2020)
- ✅ [Jester](/tools/data/jester/README_zh-CN.md) \[ [主页](https://20bn.com/datasets/jester/v1) \] (ICCV'2019)
- ✅ [GYM](/tools/data/gym/README_zh-CN.md) \[ [主页](https://sdolivia.github.io/FineGym/) \] (CVPR'2020)
- ✅ [ActivityNet](/tools/data/activitynet/README_zh-CN.md) \[ [主页](http://activity-net.org/) \] (CVPR'2015)
- ✅ [Diving48](/tools/data/diving48/README_zh-CN.md) \[ [主页](http://www.svcl.ucsd.edu/projects/resound/dataset.html) \] (ECCV'2018)

</details>

支持的时序动作检测数据集：

<details open>
<summary>(点击收起)</summary>

- ✅ [ActivityNet](/tools/data/activitynet/README_zh-CN.md) \[ [主页](http://activity-net.org/) \] (CVPR'2015)
- ✅ [THUMOS14](/tools/data/thumos14/README_zh-CN.md) \[ [主页](https://www.crcv.ucf.edu/THUMOS14/download.html) \] (THUMOS Challenge 2014)

</details>

支持的时空动作检测数据集：

<details open>
<summary>(点击收起)</summary>

- ✅ [AVA](/tools/data/ava/README_zh-CN.md) \[ [主页](https://research.google.com/ava/index.html) \] (CVPR'2018)
- 🔲 [UCF101-24](/tools/data/ucf101_24/README_zh-CN.md) \[ [主页](http://www.thumos.info/download.html) \] (CRCV-IR-12-01)
- 🔲 [JHMDB](/tools/data/jhmdb/README_zh-CN.md) \[ [主页](http://jhmdb.is.tue.mpg.de/) \] (ICCV'2013)

</details>

标记 🔲 代表对应数据集并未被完全支持，但提供相应的数据准备步骤。

## 安装

请参考 [安装指南](/docs_zh_CN/install.md) 进行安装

## 数据集准备

请参考 [数据准备](/docs_zh_CN/data_preparation.md) 了解数据集准备概况。所有支持的数据集都列于 [数据集清单](/docs_zh_CN/supported_datasets.md) 中

## 教程

请参考 [基础教程](/docs_zh_CN/getting_started.md) 了解 MMAction2 的基本使用。MMAction2也提供了其他更详细的教程:

- [如何编写配置文件](/docs_zh_CN/tutorials/1_config.md)
- [如何微调模型](/docs_zh_CN/tutorials/2_finetune.md)
- [如何增加新数据集](/docs_zh_CN/tutorials/3_new_dataset.md)
- [如何设计数据处理流程](/docs_zh_CN/tutorials/4_data_pipeline.md)
- [如何增加新模块](/docs_zh_CN/tutorials/5_new_modules.md)
- [如何导出模型为 onnx 格式](/docs_zh_CN/tutorials/6_export_model.md)
- [如何自定义模型运行参数](/docs_zh_CN/tutorials/7_customize_runtime.md)

MMAction2 也提供了相应的中文 Colab 教程，可以点击 [这里](https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial_zh-CN.ipynb) 进行体验！

## 常见问题

请参考 [FAQ](/docs_zh_CN/faq.md) 了解其他用户的常见问题

## 许可

该项目开源自 [Apache 2.0 license](/LICENSE)

## 引用

如果你觉得 MMAction2 对你的研究有所帮助，可以考虑引用它：

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

## 参与贡献

我们非常欢迎用户对于 MMAction2 做出的任何贡献，可以参考 [贡献指南](/.github/CONTRIBUTING.md) 文件了解更多细节

## 致谢

MMAction2 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。
我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/zhihu_qrcode.jpg" height="400" />  <img src="https://github.com/open-mmlab/mmaction2/raw/master/resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
