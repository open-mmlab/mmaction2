# 分析工具

除了训练/测试脚本外，我们在 `tools/` 目录下还提供了许多有用的工具。

## 分析工具链接

<!-- TOC -->

- [](#分析工具)
  - [分析工具](#分析工具)
  - [模型转换](#模型转换)
    - [准备模型进行发布](#准备模型进行发布)
  - [杂项](#杂项)
    - [评估指标](#评估指标)
    - [打印完整配置](#打印完整配置)
    - [检查视频](#检查视频)
    - [多流融合](#多流融合)

<!-- TOC -->

## 模型转换

### 准备模型进行发布

`tools/deployment/publish_model.py` 帮助用户准备他们的模型进行发布。

在将模型上传到 AWS 之前，您可能想要：

（1）将模型权重转换为 CPU 张量。
（2）删除优化器状态信息。
（3）计算权重文件的哈希值，并将哈希值添加到文件名中。

```shell
python tools/deployment/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如，

```shell
python tools/deployment/publish_model.py work_dirs/tsn_r50_8xb32-1x1x3-100e_kinetics400-rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

最终输出的文件名将是 `tsn_r50_8xb32-1x1x3-100e_kinetics400-rgb-{hash id}.pth`。

## 杂项

### 评估指标

`tools/analysis_tools/eval_metric.py` 根据配置文件评估保存在文件中的结果的某些指标。

保存的结果文件是通过在 `tools/test.py` 中设置参数 `--out ${RESULT_FILE}` 来创建的，以指示结果文件，其中存储了整个模型的最终输出。

```shell
python tools/analysis/eval_metric.py ${CONFIG_FILE} ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--cfg-options ${CFG_OPTIONS}] [--eval-options ${EVAL_OPTIONS}]
```

### 打印完整配置

`tools/analysis_tools/print_config.py` 逐字打印整个配置，展开所有导入项。

```shell
python tools/analysis_tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### 检查视频

`tools/analysis_tools/check_videos.py` 使用指定的视频编码器迭代由输入配置文件指定的所有样本，查找无效的视频（损坏或缺失），并将相应的文件路径保存到输出文件中。请注意，删除无效视频后，用户需要重新生成视频文件列表。

```shell
python tools/analysis_tools/check_videos.py ${CONFIG} [-h] [--options OPTIONS [OPTIONS ...]] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--output-file OUTPUT_FILE] [--split SPLIT] [--decoder DECODER] [--num-processes NUM_PROCESSES] [--remove-corrupted-videos]
```

### 多流融合

`tools/analysis_tools/report_accuracy.py` 使用推理保存的结果（在测试时设置 `--dump res.pkl`）来融合多流预测分数，即后融合（late fusion）。

```shell
python tools/analysis_tools/report_accuracy.py [--preds ${RESULT_PKL_1 [RESULT_PKL_2 ...]}] [--coefficients ${COEFFICIENT_1 [COEFFICIENT_2, ...]}] [--apply-softmax]
```

以 joint-bone 融合为例，这是基于骨骼动作识别任务的一种常见实践。

```shell
python tools/analysis_tools/report_accuracy.py --preds demo/fuse/joint.pkl demo/fuse/bone.pkl --coefficients 1.0 1.0
```

```
Mean Class Accuracy: 0.9180
Top 1 Accuracy: 0.9333
Top 5 Accuracy: 0.9833
```
