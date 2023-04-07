# Overview

## What is MMAction2

MMAction2 is an open source toolkit based on PyTorch, supporting numerous video understanding models, including action recognition, skeleton-based action recognition, spatio-temporal action detection and temporal action localization. In addition, it supports widely-used academic datasets and provides many useful tools, assisting users in exploring various aspects of models and datasets and implementing high-quality algorithms. Generally, it has the following features.

One-stop, Multi-model: MMAction2 supports various video understanding tasks and implements the latest models for action recognition, localization, detection.

Modular Design: MMAction2’s modular design allows users to define and reuse modules in the model on demand.

Various Useful Tools: MMAction2 provides many analysis tools, including visualizers, validation scripts, evaluators, etc., to help users troubleshoot, finetune or compare models.

Powered by OpenMMLab: Like other algorithm libraries in OpenMMLab family, MMAction2 follows OpenMMLab’s rigorous development guidelines and interface conventions, significantly reducing the learning cost of users familiar with other projects in OpenMMLab family. In addition, benefiting from the unified interfaces among OpenMMLab, you can easily call the models implemented in other OpenMMLab projects (e.g. MMClassification) in MMAction2, facilitating cross-domain research and real-world applications.

<table><tr>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/1.x/resources/mmaction2_overview.gif" width="380px">
    <p style="text-align: center;">Action Recognition</p></td>
  <td><img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="380px"><br>
    <p style="text-align: center;">Skeleton-based Action Recognition</p></td>
</table></tr>
<table><tr>
  <td><img src="https://user-images.githubusercontent.com/30782254/155710881-bb26863e-fcb4-458e-b0c4-33cd79f96901.gif" width="380px">
    <p style="text-align: center;">Spatio-Temporal Action Detection</p></td>
  <td><img src="https://github.com/open-mmlab/mmaction2/raw/1.x/resources/spatio-temporal-det.gif" width="380px"><br>
    <p style="text-align: center;">Spatio-Temporal Action Detection</p></td>
</table></tr>

## How to use the documentation

We have prepared a wealth of documents to meet your various needs:

<details open>
<summary><b>For the basic usage of MMAction2</b></summary>

- [Installation](installation.md)
- [Quick Run](quick_run.md)
- [Inference](../user_guides/Inference.md)

</details>

<details open>
<summary><b>For training on supported dataset</b></summary>

- [learn about configs](../user_guides/config.md)
- [prepare dataset](../user_guides/prepare_dataset.md)
- [Training and testing](../user_guides/train_test.md)

</details>

<details open>
<summary><b>For looking for some common issues</b></summary>

- [FAQs](faq.md)
- [Useful tools](../useful_tools.md)

</details>

<details open>
<summary><b>For a general understanding about MMAction2</b></summary>

- [20-minute tour to MMAction2](guide_to_framework.md)
- [Data flow in MMAction2](../advanced_guides/dataflow.md)

</details>

<details open>
<summary><b>For advanced usage about custom training</b></summary>

- [Customize models](../advanced_guides/customize_models.md)
- [Customize datasets](../advanced_guides/customize_dataset.md)
- [Customize data transformation and augmentation](../advanced_guides/customize_pipeline.md)
- [Customize optimizer and scheduler](../advanced_guides/customize_optimizer.md)
- [Customize logging](../advanced_guides/customize_logging.md)

</details>

<details open>
<summary><b>For supported model zoo and dataset zoo</b></summary>

- [Model Zoo](../model_zoo/modelzoo.md)
- [Dataset Zoo](../datasetzoo.md)

</details>

<details open>
<summary><b>For migration from MMAction2 0.x</b></summary>

- [Migration](../migration.md)

</details>

<details open>
<summary><b>For researchers and developers who are willing to contribute to MMAction2</b></summary>

- [Contribution Guide](contribution_guide.md)

</details>
