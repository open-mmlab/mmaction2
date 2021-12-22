除了训练/测试脚本外，MMAction2 还在 `tools/` 目录下提供了许多有用的工具。

## 目录

<!-- TOC -->

- [日志分析](#日志分析)
- [模型复杂度分析](#模型复杂度分析)
- [模型转换](#模型转换)
  - [导出 MMAction2 模型为 ONNX 格式（实验特性）](#导出-MMAction2-模型为-ONNX-格式（实验特性）)
  - [发布模型](#发布模型)
- [其他脚本](#其他脚本)
  - [指标评价](#指标评价)
  - [打印完整配置](#打印完整配置)

<!-- TOC -->

## 日志分析

输入变量指定一个训练日志文件，可通过 `tools/analysis/analyze_logs.py` 脚本绘制 loss/top-k 曲线。本功能依赖于 `seaborn`，使用前请先通过 `pip install seaborn` 安装依赖包。

![准确度曲线图](https://github.com/open-mmlab/mmaction2/raw/master/resources/acc_curve.png)

```shell
python tools/analysis/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

例如:

- 绘制某日志文件对应的分类损失曲线图。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- 绘制某日志文件对应的 top-1 和 top-5 准确率曲线图，并将曲线图导出为 PDF 文件。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys top1_acc top5_acc --out results.pdf
    ```

- 在同一图像内绘制两份日志文件对应的 top-1 准确率曲线图。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys top1_acc --legend run1 run2
    ```

    用户还可以通过本工具计算平均训练速度。

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
    ```

- 计算某日志文件对应的平均训练速度。

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time work_dirs/some_exp/20200422_153324.log.json
    ```

    预计输出结果如下所示：

    ```text
    -----Analyze train time of work_dirs/some_exp/20200422_153324.log.json-----
    slowest epoch 60, average time is 0.9736
    fastest epoch 18, average time is 0.9001
    time std over epochs is 0.0177
    average iter time: 0.9330 s/iter
    ```

## 模型复杂度分析

`/tools/analysis/get_flops.py` 是根据 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 库改编的脚本，用于计算输入变量指定模型的 FLOPs 和参数量。

```shell
python tools/analysis/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

预计输出结果如下所示：

```text
==============================
Input shape: (1, 3, 32, 340, 256)
Flops: 37.1 GMac
Params: 28.04 M
==============================
```

**注意**：该工具仍处于试验阶段，不保证该数字绝对正确。
用户可以将结果用于简单比较，但若要在技术报告或论文中采用该结果，请仔细检查。

(1) FLOPs 与输入变量形状有关，但是模型的参数量与输入变量形状无关。2D 行为识别器的默认形状为 (1, 3, 340, 256)，3D 行为识别器的默认形状为 (1, 3, 32, 340, 256)。
(2) 部分算子不参与 FLOPs 以及参数量的计算，如 GN 和一些自定义算子。更多详细信息请参考 [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py)

## 模型转换

### 导出 MMAction2 模型为 ONNX 格式（实验特性）

`/tools/deployment/pytorch2onnx.py` 脚本用于将模型转换为 [ONNX](https://github.com/onnx/onnx) 格式。
同时，该脚本支持比较 PyTorch 模型和 ONNX 模型的输出结果，验证输出结果是否相同。
本功能依赖于 `onnx` 以及 `onnxruntime`，使用前请先通过 `pip install onnx onnxruntime` 安装依赖包。
请注意，可通过 `--softmax` 选项在行为识别器末尾添加 Softmax 层，从而获取 `[0, 1]` 范围内的预测结果。

- 对于行为识别模型，请运行：

    ```shell
    python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
    ```

- 对于时序动作检测模型，请运行：

    ```shell
    python tools/deployment/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
    ```

### 发布模型

`tools/deployment/publish_model.py` 脚本用于进行模型发布前的准备工作，主要包括：

(1) 将模型的权重张量转化为 CPU 张量。
(2) 删除优化器状态信息。
(3) 计算模型权重文件的哈希值，并将哈希值添加到文件名后。

```shell
python tools/deployment/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如,

```shell
python tools/deployment/publish_model.py work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

最终，输出文件名为 `tsn_r50_1x1x3_100e_kinetics400_rgb-{hash id}.pth`。

## 其他脚本

### 指标评价

`tools/analysis/eval_metric.py` 脚本通过输入变量指定配置文件，以及对应的结果存储文件，计算某一评价指标。

结果存储文件通过 `tools/test.py` 脚本（通过参数 `--out ${RESULT_FILE}` 指定）生成，保存了指定模型在指定数据集中的预测结果。

```shell
python tools/analysis/eval_metric.py ${CONFIG_FILE} ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--cfg-options ${CFG_OPTIONS}] [--eval-options ${EVAL_OPTIONS}]
```

### 打印完整配置

`tools/analysis/print_config.py` 脚本会解析所有输入变量，并打印完整配置信息。

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```

### 检查视频

`tools/analysis/check_videos.py` 脚本利用指定视频编码器，遍历指定配置文件视频数据集中所有样本，寻找无效视频文件（文件破损或者文件不存在），并将无效文件路径保存到输出文件中。请注意，删除无效视频文件后，需要重新生成视频文件列表。

```shell
python tools/analysis/check_videos.py ${CONFIG} [-h] [--options OPTIONS [OPTIONS ...]] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]] [--output-file OUTPUT_FILE] [--split SPLIT] [--decoder DECODER] [--num-processes NUM_PROCESSES] [--remove-corrupted-videos]
```
