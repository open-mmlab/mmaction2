<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/1.x/resources/mmaction2_logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/1.x/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

[📘文档](https://mmaction2.readthedocs.io/zh_CN//1.x/) |
[🛠️安装指南](https://mmaction2.readthedocs.io/zh_CN//1.x/get_started.html) |
[👀模型库](https://mmaction2.readthedocs.io/zh_CN//1.x/modelzoo.html) |
[🆕更新](https://mmaction2.readthedocs.io/zh_CN/1.x/notes/changelog.html) |
[🚀进行中项目](https://github.com/open-mmlab/mmaction2/projects) |
[🤔问题反馈](https://github.com/open-mmlab/mmaction2/issues/new/choose)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

[English](/README.md) | 简体中文

## 简介

MMAction2 是一款基于 PyTorch 的视频理解开源工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一

1.x 分支代码目前支持 **PyTorch 1.6以上** 的版本

<div align="center">
  <div style="float:left;margin-right:10px;">
  <img src="https://github.com/open-mmlab/mmaction2/raw/1.x/resources/mmaction2_overview.gif" width="380px"><br>
    <p style="font-size:1.5vw;">Kinetics-400 上的动作识别</p>
  </div>
  <div style="float:right;margin-right:0px;">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="font-size:1.5vw;">NTURGB+D-120 上的基于人体姿态的动作识别</p>
  </div>
</div>
<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Kinetics-400 上的基于 skeleton 的时空动作检测和动作识别</p>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/1.x/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">AVA-2.1 上的时空动作检测</p>
</div>

## 主要特性

- **模块设计**：MMAction2 将统一的视频理解框架解耦成不同的模块组件，通过组合不同的模块组件，用户可以便捷地构建自定义的视频理解模型

- **支持多种任务和数据集**：MMAction2 支持多种视频理解任务，包括动作识别，时序动作检测，时空动作检测以及基于人体姿态的动作识别

- **详尽的单元测试和文档**：MMAction2 提供了详尽的说明文档，API 接口说明，全面的单元测试，以供社区参考

## 更新记录

**v1.0.0rc3 版本 (2023.02.10)**:

- 支持动作识别模型 UniFormer V1（ICLR'2022），UniFormer V2（Arxiv'2022）
- 支持训练 MViT V2（CVPR'2022）和 MaskFeat（CVPR'2022）微调
- 为 MMAction2 模型提供统一的推理接口实现视频分析任务的快速预测 ([demo](https://github.com/open-mmlab/mmaction2/blob/dev-1.x/demo/README.md#inferencer))

## 安装

MMAction2 依赖 [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine), [MMDetection](https://github.com/open-mmlab/mmdetection) (可选), [MMPose](https://github.com/open-mmlab/mmpose) (可选)，以下是安装的简要步骤。
更详细的安装指南请参考 [install.md](https://mmaction2.readthedocs.io/zh_CN/1.x/get_started.html) 。

```shell
conda create --name openmmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch  # 以上命令将自动安装最新版本的 PyTorch 和 cudatoolkit,请检查它们是否和你的环境匹配
pip install -U openmim
mim install mmengine 'mmcv>=2.0.0rc1'
mim install "mmdet>=3.0.0rc5"  # 可选
mim install "mmpose>=1.0.0rc0"  # 可选
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout 1.x
pip3 install -e .
```

## 模型库

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">行为识别方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/c3d/README.md">C3D</a> (CVPR'2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tsn/README.md">TSN</a> (ECCV'2016)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/i3d/README.md">I3D</a> (CVPR'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/c2d/README.md">C2D</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/i3d/README.md">I3D Non-Local</a> (CVPR'2018)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/r2plus1d/README.md">R(2+1)D</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/trn/README.md">TRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tsm/README.md">TSM</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tsm/README.md">TSM Non-Local</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/slowonly/README.md">SlowOnly</a> (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/slowfast/README.md">SlowFast</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/csn/README.md">CSN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tin/README.md">TIN</a> (AAAI'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tpn/README.md">TPN</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/x3d/README.md">X3D</a> (CVPR'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition_audio/resnet/README.md">MultiModality: Audio</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/tanet/README.md">TANet</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/timesformer/README.md">TimeSformer</a> (ICML'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/swin/README.md">VideoSwin</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/videomae/README.md">VideoMAE</a> (NeurIPS'2022)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/mvit/README.md">MViT V2</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/uniformer/README.md">UniFormer V1</a> (ICLR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/recognition/uniformerv2/README.md">UniFormer V2</a> (Arxiv'2022)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时序动作检测方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/localization/ssn/README.md">SSN</a> (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/localization/bsn/README.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/localization/bmn/README.md">BMN</a> (ICCV'2019)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/detection/acrn/README.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/detection/ava/README.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/detection/ava/README.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/detection/lfb/README.md">LFB</a> (CVPR'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的动作识别方法</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/skeleton/stgcn/README.md">ST-GCN</a> (AAAI'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/skeleton/2s-agcn/README.md">2s-AGCN</a> (CVPR'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/skeleton/posec3d/README.md">PoseC3D</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/configs/skeleton/stgcnpp/README.md">STGCN++</a> (ArXiv'2022)</td>
    <td></td>
  </tr>
</table>

各个模型的结果和设置都可以在对应的 config 目录下的 *README_zh-CN.md* 中查看。整体的概况也可也在 [**模型库**](https://mmaction2.readthedocs.io/zh_CN/1.x/modelzoo.html) 页面中查看。

MMAction2 将跟进学界的最新进展，并支持更多算法和框架。如果您对 MMAction2 有任何功能需求，请随时在 [问题](https://github.com/open-mmlab/mmaction2/issues/19) 中留言。

## 数据集

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">动作识别数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/hmdb51/README.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">Homepage</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/ucf101/README.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/kinetics/README.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">Homepage</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/sthv1/README.md">SthV1</a>  (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/sthv2/README.md">SthV2</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/something-something">Homepage</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/diving48/README.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">Homepage</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/jester/README.md">Jester</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/jester">Homepage</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/mit/README.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">Homepage</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/mmit/README.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">Homepage</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/hvu/README.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">Homepage</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/omnisource/README.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">Homepage</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/gym/README.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">Homepage</a>) (CVPR'2020)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时序动作检测数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/thumos14/README.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">Homepage</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时空动作检测数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/ucf101_24/README.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/jhmdb/README.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">Homepage</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/ava/README.md">AVA</a> (<a href="https://research.google.com/ava/index.html">Homepage</a>) (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/ava_kinetics/README.md">AVA-Kinetics</a> (<a href="https://research.google.com/ava/index.html">Homepage</a>) (Arxiv'2020)</td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">基于骨骼点的动作识别数据集</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/1.x/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
  </tr>
</table>

标记 * 代表对应数据集并未被完全支持，但提供相应的数据准备步骤。整体的概况也可也在 [**数据集**](https://mmaction2.readthedocs.io/zh_CN/latest/supported_datasets.html) 页面中查看。

## 数据集准备

请参考 [数据准备](https://mmaction2.readthedocs.io/en/1.x/user_guides/2_data_prepare.html) 了解数据集准备概况。所有支持的数据集都列于 [数据集清单](https://mmaction2.readthedocs.io/zh_CN/latest/supported_datasets.html) 中。

## FAQ

请参考 [FAQ](docs/zh_cn/notes/faq.md) 了解其他用户的常见问题。

## 相关工作

目前有许多研究工作或工程项目基于 MMAction2 搭建，例如：

- Video Swin Transformer. [\[论文\]](https://arxiv.org/abs/2106.13230)[\[代码\]](https://github.com/SwinTransformer/Video-Swin-Transformer)
- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [\[论文\]](https://arxiv.org/abs/2107.10161)[\[代码\]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [\[论文\]](https://arxiv.org/abs/2103.17263)[\[代码\]](https://github.com/xvjiarui/VFS)

更多详情可见 [相关工作](docs/en/notes/projects.md) 。

## 许可

该项目开源自 [Apache 2.0 license](LICENSE).

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

我们非常欢迎用户对于 MMAction2 做出的任何贡献，可以参考 [贡献指南](https://github.com/open-mmlab/mmcv/blob/2.x/CONTRIBUTING_zh-CN.md) 文件了解更多细节。

## 致谢

MMAction2 是一款由不同学校和公司共同贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。
我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱和基准测试
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
