<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_logo.png" width="600"/>
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

[![Documentation](https://readthedocs.org/projects/mmaction2/badge/?version=latest)](https://mmaction2.readthedocs.io/en/latest/)
[![actions](https://github.com/open-mmlab/mmaction2/workflows/build/badge.svg)](https://github.com/open-mmlab/mmaction2/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmaction2/branch/main/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmaction2)
[![PyPI](https://img.shields.io/pypi/v/mmaction2)](https://pypi.org/project/mmaction2/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/blob/main/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmaction2.svg)](https://github.com/open-mmlab/mmaction2/issues)

[📘中文文档](https://mmaction2.readthedocs.io/zh_CN/latest/index.html) |
[🛠️安装指南](https://mmaction2.readthedocs.io/zh_CN/latest/get_started/installation.html) |
[👀模型库](https://mmaction2.readthedocs.io/zh_CN/latest/modelzoo_statistics.html) |
[🆕更新日志](https://mmaction2.readthedocs.io/en/latest/notes/changelog.html) |
[🚀进行中项目](https://github.com/open-mmlab/mmaction2/projects) |
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
- [🥳 🚀 最新进展](#--最新进展-)
- [📖 简介](#-简介-)
- [🎁 主要功能](#-主要功能-)
- [🛠️ 安装](#️-安装-)
- [👀 模型库](#-模型库-)
- [👨‍🏫 新手入门](#-新手入门-)
- [🎫 许可证](#-许可证-)
- [🖊️ 引用](#️-引用-)
- [🙌 参与贡献](#-参与贡献-)
- [🤝 致谢](#-致谢-)
- [🏗️ OpenMMLab 的其他项目](#️-openmmlab-的其他项目-)
- [❤️ 欢迎加入 OpenMMLab 社区](#️-欢迎加入-openmmlab-社区-)

## 🥳 🚀 最新进展 [🔝](#-table-of-contents)

**默认分支已经从 `master` （当前的`0.x`） 切换到 `main`（之前的 `1.x`），我们建议用户更新至最新版本，其支持更多模型，更强的预训练权重，以及更简洁的代码实现。详情请参阅[迁移指南](https://mmaction2.readthedocs.io/zh_cn/latest/migration.html)**

**Release (2023.07.04)**: v1.1.0 支持以下新功能:

- 支持基于 CLIP 的多模态模型: ActionCLIP(Arxiv'2021) 和 CLIP4clip(ArXiv'2022)
- 支持丰富的 project: 手势识别, 时空行为检测 tutorial, 以及基于 [MMRazor](https://github.com/open-mmlab/mmrazor) 的知识蒸馏
- 支持 HACS-segments 数据集(ICCV'2019), MultiSports 数据集(ICCV'2021), Kinetics-710 数据集(Arxiv'2022)
- 支持 VideoMAE V2(CVPR'2023), VideoMAE(NeurIPS'2022) 支持时空行为检测任务
- 支持 TCANet(CVPR'2021)
- 支持 [纯 Python 风格的配置文件](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta) 和使用 MIM 一键下载数据集

## 📖 简介 [🔝](#-table-of-contents)

MMAction2 是一款基于 PyTorch 开发的行为识别开源工具包，是 [open-mmlab](https://github.com/open-mmlab)  项目的一个子项目。

<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px">
  <p style="font-size:1.5vw;"> Kinetics-400 数据集行为识别结果（左） 和 NTU-RGB+D-120 数据集基于骨架的行为识别结果（右）</p>
</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="580px"/><br>
    <p style="font-size:1.5vw;">Kinetics-400 数据集基于骨骼点的时空行为检测及视频行为识别结果</p>
</div>
<div align="center">
  <img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="800px"/><br>
    <p style="font-size:1.5vw;">AVA-2.1 数据集时空行为检测结果</p>
</div>

## 🎁 主要功能 [🔝](#-table-of-contents)

- **模块化设计**： 我们将视频理解框架拆分成了不同模块，用户可以很方便地通过组合不同的模块来构建出自定义的视频理解框架。

- **支持五种主要的视频理解任务**： MMAction2 为视频理解任务实现了多种多样的算法，包括行为识别，时序动作定位，时空动作检测，基于骨骼点的行为识别，以及视频检索。

- **详尽的单元测试和文档**：我们提供了详尽的文档和 API 参考手册，以及单元测试。

## 🛠️ 安装 [🔝](#-table-of-contents)

MMAction2依赖于 [PyTorch](https://pytorch.org/)，[MMCV](https://github.com/open-mmlab/mmcv)，[MMEngine](https://github.com/open-mmlab/mmengine)，[MMDetection](https://github.com/open-mmlab/mmdetection) （可选）和 [MMPose](https://github.com/open-mmlab/mmpose) （可选）

具体步骤请参考 [安装文档](https://mmaction2.readthedocs.io/zh_cn/latest/get_started/installation.html)。

<details close>
<summary>快速安装</summary>

```shell
conda create --name openmmlab python=3.8 -y
conda activate open-mmlab
conda install pytorch torchvision -c pytorch  # 该命令将自动安装最新版的 PyTorch 和 cudatoolkit，请确认此是否匹配你的当前环境。
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

结果及模型位于[模型库](https://mmaction2.readthedocs.io/zh_cn/latest/modelzoo_statistics.html)

<details close>

<summary>模型支持</summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">行为识别</td>
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
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/README.md">ActionCLIP</a> (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/README.md">VideoSwin</a> (CVPR'2022)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomae/README.md">VideoMAE</a> (NeurIPS'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/mvit/README.md">MViT V2</a> (CVPR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformer/README.md">UniFormer V1</a> (ICLR'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md">UniFormer V2</a> (Arxiv'2022)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomaev2/README.md">VideoMAE V2</a> (CVPR'2023)</td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时序动作定位</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/localization/bsn/README.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/localization/bmn/README.md">BMN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/localization/tcanet/README.md">TCANet</a> (CVPR'2021)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">时空行为检测</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/acrn/README.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/slowonly/README.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/slowfast/README.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/detection/lfb/README.md">LFB</a> (CVPR'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomae/README.md">VideoMAE</a> (NeurIPS'2022)</td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">基于骨骼点的行为识别</td>
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
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">视频检索</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/configs/retrieval/clip4clip/README.md">CLIP4Clip</a> (ArXiv'2022)</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

</table>

</details>

<details close>

<summary>数据集支持</summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">行为识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hmdb51/README.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">官网</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101/README.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">官网</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">官网</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">官网</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv1/README.md">SthV1</a>  (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/sthv2/README.md">SthV2</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/something-something">官网</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/diving48/README.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">官网</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/jester/README.md">Jester</a> (<a href="https://developer.qualcomm.com/software/ai-datasets/jester">官网</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/mit/README.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">官网</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/mmit/README.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">官网</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hvu/README.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">官网</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/omnisource/README.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">官网</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/gym/README.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">官网</a>) (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics710/README.md">Kinetics-710</a> (<a href="https://arxiv.org/pdf/2211.09552.pdf">官网</a>) (Arxiv'2022)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时序动作定位</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/thumos14/README.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">官网</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">官网</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hacs/README.md">HACS</a> (<a href="https://github.com/hangzhaomit/HACS-dataset">官网</a>) (ICCV'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">时空行为检测</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101_24/README.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">官网</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/jhmdb/README.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">官网</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava/README.md">AVA</a> (<a href="https://research.google.com/ava/index.html">官网</a>) (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ava_kinetics/README.md">AVA-Kinetics</a> (<a href="https://research.google.com/ava/index.html">官网</a>) (Arxiv'2020)</td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">基于骨架的行为识别</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">官网</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">官网</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">官网</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">官网</a>) (ArXiv'2021)</td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">视频检索</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/main/tools/data/video_retrieval/README.md">MSRVTT</a> (<a href="https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/">官网</a>) (CVPR'2016)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

</details>

## 👨‍🏫 新手入门 [🔝](#-table-of-contents)

我们提供了一系列简明的教程，帮助新用户轻松上手使用：

- [从 MMAction2 0.X 迁移](https://mmaction2.readthedocs.io/zh_cn/latest/migration.html)
- [学习配置相关知识](https://mmaction2.readthedocs.io/zh_cn/latest/user_guides/config.html)
- [准备数据集](https://mmaction2.readthedocs.io/zh_cn/latest/user_guides/prepare_dataset.html)
- [使用现有模型进行推理](https://mmaction2.readthedocs.io/zh_cn/latest/user_guides/inference.html)
- [训练与测试](https://mmaction2.readthedocs.io/zh_cn/latest/user_guides/train_test.html)

<details close>
<summary>基于 MMAction2 的社区工作</summary>

- Video Swin Transformer. [\[paper\]](https://arxiv.org/abs/2106.13230)[\[github\]](https://github.com/SwinTransformer/Video-Swin-Transformer)
- Evidential Deep Learning for Open Set Action Recognition, ICCV 2021 **Oral**. [\[paper\]](https://arxiv.org/abs/2107.10161)[\[github\]](https://github.com/Cogito2012/DEAR)
- Rethinking Self-supervised Correspondence Learning: A Video Frame-level Similarity Perspective, ICCV 2021 **Oral**. [\[paper\]](https://arxiv.org/abs/2103.17263)[\[github\]](https://github.com/xvjiarui/VFS)

</details>

## 🎫 许可证 [🔝](#-table-of-contents)

本项目基于 [Apache 2.0 license](LICENSE) 发布。

## 🖊️ 引用 [🔝](#-table-of-contents)

如你发现本项目对你的研究有帮助，请参考如下 bibtex 引用 MMAction2。

```BibTeX
@misc{2020mmaction2,
    title={OpenMMLab's Next Generation Video Understanding Toolbox and Benchmark},
    author={MMAction2 Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmaction2}},
    year={2020}
}
```

## 🙌 参与贡献 [🔝](#-table-of-contents)

我们感谢所有的贡献者为改进和提升 MMAction2 所作出的努力。请参考[贡献指南](https://github.com/open-mmlab/mmcv/blob/2.x/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 🤝 致谢 [🔝](#-table-of-contents)

MMAction2 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望此工具箱可以帮助大家来复现已有的方法和开发新的方法，从而为研究社区贡献力量。

## 🏗️ OpenMMLab 的其他项目 [🔝](#-table-of-contents)

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab 深度学习预训练工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMagic](https://github.com/open-mmlab/mmagic): OpenMMLab 新一代人工智能内容生成（AIGC）工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架
- [Playground](https://github.com/open-mmlab/playground): 收集和展示 OpenMMLab 相关的前沿、有趣的社区项目

## ❤️ 欢迎加入 OpenMMLab 社区 [🔝](#-table-of-contents)

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，扫描下方微信二维码添加喵喵好友，进入 MMAction2 微信交流社群。【加好友申请格式：研究方向+地区+学校/公司+姓名】

<div align="center">
<img src="./resources/zhihu_qrcode.jpg" height="400"/> <img src="./resources/miaomiao_qrcode.jpg" height="400"/>
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
