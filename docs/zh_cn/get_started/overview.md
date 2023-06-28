# 概述

## 什么是 MMAction2

MMAction2 是一个基于 PyTorch 的开源工具包，支持了大量的视频理解模型，包括**行为识别、基于骨架的行为识别、时空行为检测和时序动作定位**等多个主要方向。它还支持了大多数流行的学术数据集，并提供了许多实用工具帮助用户对数据集和模型进行多方面的探索和调试。它具有以下特点：

**全流程，多模型**：MMAction2 支持各种视频理解任务，实现了最先进的行为识别、定位、检测模型。

**模块化设计**：MMAction2 的模块化设计使用户可以根据需要定义和重用模型中的模块。

**实用工具众多**：MMAction2 提供了一系列的分析工具，如可视化器、验证脚本、评估器等，以帮助用户进行故障排除、微调或比较模型。

**由 OpenMMLab 强力驱动**：与家族内的其它算法库一样，MMAction2 遵循着 OpenMMLab 严谨的开发准则和接口约定，极大地降低了用户切换各算法库时的学习成本。同时，MMAction2 也可以非常便捷地与家族内其他算法库跨库联动，从而满足用户跨领域研究和落地的需求。

<table><tr>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
    <p style="text-align: center;">行为识别</p></td>
  <td><img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="text-align: center;">基于骨架的行为识别</p></td>
</table></tr>
<table><tr>
  <td><img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="380px">
    <p style="text-align: center;">时空动作检测</p></td>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="380px"><br>
    <p style="text-align: center;">时空动作检测</p></td>
</table></tr>

## 如何使用文档

针对不同类型的用户，我们准备了详细的指南：

<details open>
<summary><b> MMAction2 的基础用法</b></summary>

- [安装](installation.md)
- [快速运行](quick_run.md)
- [利用现有模型进行推理](../user_guides/inference.md)

</details>

<details open>
<summary><b>关于在已支持的数据集上进行训练</b></summary>

- [了解配置文件](../user_guides/config.md)
- [准备数据集](../user_guides/prepare_dataset.md)
- [训练与测试](../user_guides/train_test.md)

</details>

<details open>
<summary><b>关于使用过程中的常见问题</b></summary>

- [常见问题解答](faq.md)
- [有用的工具](../useful_tools.md)

</details>

<details open>
<summary><b>关于 MMAction2 的框架设计</b></summary>

- [20分钟 MMAction2 框架指南](guide_to_framework.md)
- [MMAction2 中的数据流](../advanced_guides/dataflow.md)

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

- [模型库](../modelzoo_statistics.md)
- [数据集](../datasetzoo_statistics.md)

</details>

<details open>
<summary><b>关于从 MMAction2 0.x 迁移</b></summary>

- [从 MMAction2 0.x 迁移](../migration.md)

</details>

<details open>
<summary><b>对于希望加入开源社区，向 MMAction2 贡献代码的研究者和开发者</b></summary>

- [如何为 MMAction2 做出贡献](contribution_guide.md)

</details>
