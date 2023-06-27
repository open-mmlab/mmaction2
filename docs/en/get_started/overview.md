# Overview

## What is MMAction2

MMAction2 is an open source toolkit based on PyTorch, supporting numerous video understanding models, including **action recognition, skeleton-based action recognition, spatio-temporal action detection and temporal action localization**. Moreover, it supports widely-used academic datasets and offers many useful tools, assisting users in exploring various aspects of models and datasets, as well as implementing high-quality algorithms. Generally, the toolkit boasts the following features:

**One-stop, Multi-model**: MMAction2 supports various video understanding tasks and implements state-of-the-art models for action recognition, localization, detection.

**Modular Design**: The modular design of MMAction2 enables users to define and reuse modules in the model as required.

**Various Useful Tools**: MMAction2 provides an array of analysis tools, such as visualizers, validation scripts, evaluators, etc., to aid users in troubleshooting, fine-tuning, or comparing models.

**Powered by OpenMMLab**: Similar to other algorithm libraries in the OpenMMLab family, MMAction2 adheres to OpenMMLab's rigorous development guidelines and interface conventions, considerably reducing the learning cost for users familiar with other OpenMMLab projects. Furthermore, due to the unified interfaces among OpenMMLab projects, it is easy to call models implemented in other OpenMMLab projects (such as MMClassification) in MMAction2, which greatly facilitates cross-domain research and real-world applications.

<table><tr>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/mmaction2_overview.gif" width="380px">
    <p style="text-align: center;">Action Recognition</p></td>
  <td><img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="text-align: center;">Skeleton-based Action Recognition</p></td>
</table></tr>
<table><tr>
  <td><img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="380px">
    <p style="text-align: center;">Spatio-Temporal Action Detection</p></td>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/main/resources/spatio-temporal-det.gif" width="380px"><br>
    <p style="text-align: center;">Spatio-Temporal Action Detection</p></td>
</table></tr>

## How to use the documentation

We have prepared a wealth of documents to meet your various needs:

<details open>
<summary><b>For the basic usage of MMAction2</b></summary>

- [Installation](installation.md)
- [Quick Run](quick_run.md)
- [Inference with existing models](../user_guides/inference.md)

</details>

<details open>
<summary><b>For training on supported dataset</b></summary>

- [Learn about Configs](../user_guides/config.md)
- [Prepare Dataset](../user_guides/prepare_dataset.md)
- [Training and Test](../user_guides/train_test.md)

</details>

<details open>
<summary><b>For looking for some common issues</b></summary>

- [FAQ](faq.md)
- [Useful tools](../useful_tools.md)

</details>

<details open>
<summary><b>For a general understanding about MMAction2</b></summary>

- [A 20-Minute Guide to MMAction2 FrameWork](guide_to_framework.md)
- [Dataflow in MMAction2](../advanced_guides/dataflow.md)

</details>

<details open>
<summary><b>For advanced usage about custom training</b></summary>

- [Customize Model](../advanced_guides/customize_models.md)
- [Customize Dataset](../advanced_guides/customize_dataset.md)
- [Customize Data Pipeline](../advanced_guides/customize_pipeline.md)
- [Customize Optimizer](../advanced_guides/customize_optimizer.md)
- [Customize Logging](../advanced_guides/customize_logging.md)

</details>

<details open>
<summary><b>For supported model zoo and dataset zoo</b></summary>

- [Model Zoo](../modelzoo_statistics.md)
- [Dataset Zoo](../datasetzoo_statistics.md)

</details>

<details open>
<summary><b>For migration from MMAction2 0.x</b></summary>

- [Migration](../migration.md)

</details>

<details open>
<summary><b>For researchers and developers who are willing to contribute to MMAction2</b></summary>

- [How to contribute to MMAction2](contribution_guide.md)

</details>
