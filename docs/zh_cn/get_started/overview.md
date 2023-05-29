# 概述

## 什么是MMAction2

MMAction2是一个基于PyTorch的开源工具包，支持大量的视频理解模型，包括**动作识别、基于骨架的动作识别、时空动作检测和时间动作定位**。此外，它支持广泛使用的学术数据集，并提供许多有用的工具，帮助用户探索模型和数据集的各个方面，以及实现高质量的算法。一般来说，该工具包具有以下特性：

**一站式、多模型**：MMAction2支持各种视频理解任务，实现最先进的动作识别、定位、检测模型。

**模块化设计**：MMAction2的模块化设计使用户可以根据需要定义和重用模型中的模块。

**各种有用的工具**：MMAction2提供了一系列的分析工具，如可视化器、验证脚本、评估器等，以帮助用户进行故障排除、微调或比较模型。

**由OpenMMLab提供支持**：与OpenMMLab家族中的其他算法库类似，MMAction2遵循OpenMMLab的严格开发指南和接口约定，大大降低了熟悉其他OpenMMLab项目的用户的学习成本。此外，由于OpenMMLab项目之间的统一接口，在MMAction2中很容易调用其他OpenMMLab项目实现的模型(如MMClassification)，这极大地便利了跨领域研究和实际应用。

<table><tr>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
    <p style="text-align: center;">动作识别</p></td>
  <td><img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="text-align: center;">基于骨架的动作识别</p></td>
</table></tr>
<table><tr>
  <td><img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="380px">
    <p style="text-align: center;">时空动作检测</p></td>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="380px"><br>
    <p style="text-align: center;">时空动作检测</p></td>
</table></tr>

## 如何使用文档

我们准备了丰富的文件来满足您的各种需求:


<details open>
<summary><b>关于MMAction2的基本用法</b></summary>

- [安装](installation.md)
- [快速运行](quick_run.md)
- [利用现有模型进行推理](../user_guides/3_inference.md)

</details>

<details open>
<summary><b>关于在支持的数据集上进行训练</b></summary>

- [了解配置文件](../user_guides/1_config.md)
- [准备数据集](../user_guides/2_data_prepare.md)
- [训练与测试](../user_guides/train_test.md)

</details>

<details open>
<summary><b>关于寻找一些常见问题</b></summary>

- [常见问题解答](faq.md)
- [有用的工具](../useful_tools.md)

</details>

<details open>
<summary><b>关于对MMAction2的一个大致了解</b></summary>

- [20分钟MMAction2框架指南](guide_to_framework.md)
- [MMAction2中的数据流](../advanced_guides/dataflow.md)

</details>

<details open>
<summary><b>关于自定义训练的高级用法</b></summary>

- [自定义模型](../advanced_guides/customize_models.md)
- [自定义数据集](../advanced_guides/customize_dataset.md)
- [自定义数据管道](../advanced_guides/customize_pipeline.md)
- [自定义优化器](../advanced_guides/customize_optimizer.md)
- [自定义日志记录](../advanced_guides/customize_logging.md)

</details>

<details open>
<summary><b>关于支持的模型库和数据集</b></summary>

- [模型库](../model_zoo/modelzoo.md)
- [数据集](../datasetzoo.md)

</details>

<details open>
<summary><b>关于从MMAction2 .x迁移</b></summary>

- [从 MMAction2 0.x 迁移](../migration.md)

</details>

<details open>
<summary><b>关于愿意为MMAction2做出贡献的研究人员和开发者</b></summary>

- [如何为MMAction2做出贡献](contribution_guide.md)

</details>
