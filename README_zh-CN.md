<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">火</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">尝试一下</font></i>
      </a>
    </sup>
  </div>

[![文档](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/zh_CN/latest/index.html)
[![构建](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![代码覆盖率](https://codecov.io/gh/open-mmlab/mmaction2/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![许可](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/main/LICENSE)
[![问题解决平均耗时](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![尚未解决问题占比](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

[📘文档](https://mmaction2.readthedocs.io/zh_CN/latest/index.html) |
[🛠️安装](https://mmaction2.readthedocs.io/zh_CN/latest/get_started.html#id2) |
[👀模型库](https://mmaction2.readthedocs.io/zh_CN/latest/modelzoo.html) |
[🆕更新日志](https://mmaction2.readthedocs.io/zh_CN/latest/notes/changelog.html) |
[🚀正在进行的项目](https://github.com/open-mmlab/mmaction2/projects) |
[🤔报告问题](https://github.com/open-mmlab/mmaction2/issues/new/choose)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

[English](/README.md) | 简体中文

## 📄 目录

- [📄 目录](#-目录)
- [🥳 🚀 最新进展 🔝](#--最新进展-)
- [📖 简介 🔝](#-简介-)
- [🎁 主要功能 🔝](#-主要功能-)
- [🛠️ 安装 🔝](#️-安装-)
- [👀 模型库 🔝](#-模型库-)
- [👨‍🏫 入门 🔝](#-入门-)
- [🎫 许可 🔝](#-许可-)
- [🖊️ 引用 🔝](#️-引用-)
- [🙌 参与贡献 🔝](#-参与贡献-)
- [🤝 致谢 🔝](#-致谢-)
- [🏗️ OpenMMLab所有项目 🔝](#️-openmmlab所有项目-)

## 🥳 🚀 最新进展 [🔝](#-table-of-contents)

**默认分支将从`master`（以前的`0.x`）转成`main`（当前的`1.x`），我们建议用户更新至最新版本，其支持更多模型，更强的预训练检查点，以及更简洁的代码实现。详情请参阅[迁移指南](https://mmaction2.readthedocs.io/en/latest/migration.html)**

**发布(2023.04.06)**: v1.0.0支持以下新功能:

- 支持RGB-PoseC3D(CVPR'2022)。
- 支持训练UniFormer V2(Arxiv'2022)
- 支持MSG3D(CVPR'2020)和CTRGCN(CVPR'2021)
- 优化并增加更友好的文档

## 📖 简介 [🔝](#-table-of-contents)

MMAction2是一款基于PyTorch开发的动作识别开源工具包，是[open-mmlab](https://github.com/open-mmlab) 项目的一个子项目。

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px">
  <p style="font-size:1.5vw;"> Kinetics-400中动作识别（左） 和 NTU-RGB+D-120中基于骨架的动作识别（右）</p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Kinetics-400中基于骨架的时空动作检测及识别结果</p>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">AVA-2.1中时空动作检测结果</p>
</div>

## 🎁 主要功能 [🔝](#-table-of-contents)

- **模块化设计**： 我们将视频理解框架拆分成了不同模块，用户可以很方便地通过组合不同的模块来构建出自定义化的视频理解框架。

- **支持四种主要的视频理解任务**： MMAction2为视频理解任务实现了多种多样的算法，包括动作识别，动作定位，时空动作检测，以及基于骨架的动作检测。

- **严密的测试和丰富的文档**：我们提供了详尽的文档和API参考手册，以及单元测试。

## 🛠️ 安装 [🔝](#-table-of-contents)

MMAction2依赖于[PyTorch](https://pytorch.org/)，[MMCV](https://github.com/open-mmlab/mmcv)，[MMEngine](https://github.com/open-mmlab/mmengine)，[MMDetection](https://github.com/open-mmlab/mmdetection) (可选)和[MMPose](https://github.com/open-mmlab/mmpose) (可选)
具体步骤请参见[install.md](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)

<details close>
<summary>快速上手</summary>

```shell
conda create --name openmmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch  # 该命令将自动安装最新版的PyTorch和cudatoolkit，请确认此是否匹配你的当前环境。
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # 可选
mim install mmpose  # 可选
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

</details>

## 👀 模型库 [🔝](#-table-of-contents)

结果及模型位于[模型集](https://mmaction2.readthedocs.io/en/latest/model_zoo/modelzoo.html)

<details close>

<summary>支持的模型</summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">动作识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c3d/README.md">C3D</a> (CVPR'2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/README.md">TSN</a> (ECCV'2016)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/i3d/README.md">I3D</a> (CVPR'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/c2d/README.md">C2D</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/i3d/README.md">I3D Non-Local</a> (CVPR'2018)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/r2plus1d/README.md">R(2+1)D</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/trn/README.md">TRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/README.md">TSM</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsm/README.md">TSM Non-Local</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/slowonly/README.md">SlowOnly</a> (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/slowfast/README.md">SlowFast</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/csn/README.md">CSN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tin/README.md">TIN</a> (AAAI'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tpn/README.md">TPN</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/x3d/README.md">X3D</a> (CVPR'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition_audio/resnet/README.md">MultiModality: Audio</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tanet/README.md">TANet</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/timesformer/README.md">TimeSformer</a> (ICML'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/README.md">VideoSwin</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomae/README.md">VideoMAE</a> (NeurIPS'2022)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/mvit/README.md">MViT V2</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformer/README.md">UniFormer V1</a> (ICLR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md">UniFormer V2</a> (Arxiv'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomaev2/README.md">VideoMAE V2</a> (CVPR'2023)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">动作定位</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/localization/bsn/README.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/localization/bmn/README.md">BMN</a> (ICCV'2019)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空动作检测</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/acrn/README.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/slowonly/README.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/slowfast/README.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/lfb/README.md">LFB</a> (CVPR'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨架的动作识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcn/README.md">ST-GCN</a> (AAAI'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/2s-agcn/README.md">2s-AGCN</a> (CVPR'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/posec3d/README.md">PoseC3D</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcnpp/README.md">STGCN++</a> (ArXiv'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/projects/ctrgcn/README.md">CTRGCN</a> (CVPR'2021)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/projects/msg3d/README.md">MSG3D</a> (CVPR'2020)</td>
    <td></td>
  </tr>
</table>

</details>

<details close>

<summary>支持数据集</summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">动作识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hmdb51/README.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">主页</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101/README.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">主页</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">主页</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">主页</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv1/README.md">SthV1</a>  (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv2/README.md">SthV2</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/something-something">主页</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/diving48/README.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">主页</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/jester/README.md">Jester</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/jester">主页</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/mit/README.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">主页</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/mmit/README.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">主页</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hvu/README.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">主页</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/omnisource/README.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">主页</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/gym/README.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">主页</a>) (CVPR'2020)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">动作定位</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/thumos14/README.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">主页</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">主页</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时空动作检测</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101_24/README.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">主页</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/jhmdb/README.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">主页</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava/README.md">AVA</a> (<a href="https://research.google.com/ava/index.html">主页</a>) (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava_kinetics/README.md">AVA-Kinetics</a> (<a href="https://research.google.com/ava/index.html">主页</a>) (Arxiv'2020)</td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">基于骨架的动作识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">主页</a>) (ArXiv'2021)</td>
  </tr>
</table>

</details>

## 👨‍🏫 入门 [🔝](#-table-of-contents)

对于基本用法，我们提供了如下用户指南：

- [从MMAction2 0.X版迁移过来](https://mmaction2.readthedocs.io/en/latest/migration.html)
- [学习配置相关知识](https://mmaction2.readthedocs.io/en/latest/user_guides/config.html#)
- [准备数据集](https://mmaction2.readthedocs.io/en/latest/user_guides/prepare_dataset.html)
- [使用现有模型进行推理](https://mmaction2.readthedocs.io/en/latest/user_guides/inference.html)
- [训练与测试](https://mmaction2.readthedocs.io/en/latest/user_guides/train_test.html)

<details close>
<summary>社区用户的基于MMAction2的研究工作</summary>

- Video Swin Transformer. [\[paper\]](https://arxiv.org/abs/2106.13230)[\[github\]](https://github.com/SwinTransformer/Video-Swin-Transformer)
- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [\[paper\]](https://arxiv.org/abs/2107.10161)[\[github\]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [\[paper\]](https://arxiv.org/abs/2103.17263)[\[github\]](https://github.com/xvjiarui/VFS)

</details>

## 🎫 许可 [🔝](#-table-of-contents)

本项目基于[Apache 2.0 license](LICENSE)发布。

## 🖊️ 引用 [🔝](#-table-of-contents)

如你发现本项目对你的研究有帮助，请考虑引用：

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

## 🙌 参与贡献 [🔝](#-table-of-contents)

我们感谢所有参与改进MMAction2的贡献，更多贡献指南请参阅MMCV下的[CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/2.x/CONTRIBUTING.md)

## 🤝 致谢 [🔝](#-table-of-contents)

MMAction2是一个来自众多高校和企业的研究人员及工程师共同参与构建的开源项目，我们感谢所有实现他们的算法，添加新功能的贡献者，以及那些提供有价值的反馈的用户。我们希望通过提供一款灵活的能够实现现有算法以及开发新的模型工具包，来服务于日益增长的研究群体，

## 🏗️ OpenMMLab所有项目 [🔝](#-table-of-contents)

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab pre-training toolbox and benchmark.
- [MMagic](https://github.com/open-mmlab/mmagic): Open**MM**Lab **A**dvanced, **G**enerative and **I**ntelligent **C**reation toolbox.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [Playground](https://github.com/open-mmlab/playground): A central hub for gathering and showcasing amazing projects built upon OpenMMLab.
