除了训练/测试脚本外，MMAction2 还在 `tools/` 目录下提供了许多有用的工具。

## 目录

<!-- TOC -->

- [日志分析](#日志分析)
- [模型复杂度分析](#模型复杂度分析)
- [模型转换](#模型转换)
  - [导出 MMAction2 模型为 ONNX 格式（试验阶段）](#导出-MMAction2-模型为-ONNX-格式（试验阶段）)
  - [发布模型](#发布模型)
- [其他脚本](#其他脚本)
  - [指标评价](#指标评价)
  - [打印完整配置](#打印完整配置)

<!-- TOC -->

## 日志分析

给定一个训练日志文件，可通过 `tools/analysis/analyze_logs.py` 脚本绘制 loss/top-k 曲线。要使用这个功能，要先通过 `pip install seaborn` 安装所需的依赖包。

![acc_curve_image](/docs/imgs/acc_curve.png)

```shell
python tools/analysis/analyze_logs.py plot_curve ${JSON_LOGS} [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

例如:

- 绘制某结果的分类损失图。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys loss_cls --legend loss_cls
    ```

- 绘制某结果的 top-1 和 top-5 准确率图像，并将其导出为 PDF 文件。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log.json --keys top1_acc top5_acc --out results.pdf
    ```

- 在同一图像内绘制两份结果文件的 top-1 准确率。

    ```shell
    python tools/analysis/analyze_logs.py plot_curve log1.json log2.json --keys top1_acc --legend run1 run2
    ```

    You can also compute the average training speed.

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time ${JSON_LOGS} [--include-outliers]
    ```

- 计算某个配置文件的平均训练速度

    ```shell
    python tools/analysis/analyze_logs.py cal_train_time work_dirs/some_exp/20200422_153324.log.json
    ```

    预计结果输出如下：

    ```text
    -----Analyze train time of work_dirs/some_exp/20200422_153324.log.json-----
    slowest epoch 60, average time is 0.9736
    fastest epoch 18, average time is 0.9001
    time std over epochs is 0.0177
    average iter time: 0.9330 s/iter
    ```

## 模型复杂度分析

`/tools/analysis/get_flops.py` 文件是根据 [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch) 库改编的脚本，用于计算给定模型的 FLOPs 和参数量。

```shell
python tools/analysis/get_flops.py ${CONFIG_FILE} [--shape ${INPUT_SHAPE}]
```

预计结果输出如下：

```text
==============================
Input shape: (1, 3, 32, 340, 256)
Flops: 37.1 GMac
Params: 28.04 M
==============================
```

**注意**：该工具仍处于试验阶段，不保证该数字绝对正确。
用户可以将结果用于简单比较，但若要在技术报告或论文中采用该结果之前，请仔细检查。

(1) FLOPs 与输入变量的形状有关，但是模型的参数量与输入变量的形状无关。2D 行为识别器的默认形状为 (1, 3, 340, 256)，3D 行为识别器的默认形状为 (1, 3, 32, 340, 256)。
(2) 某些算子没被列入计算中，如 GN 和一些自定义算子。更多详细信息请参考 [`mmcv.cnn.get_model_complexity_info()`](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py)

## 模型转换

### 导出 MMAction2 模型为 ONNX 格式（试验阶段）

`/tools/pytorch2onnx.py` 脚本用于将模型转换为 [ONNX](https://github.com/onnx/onnx) 格式。
它同时也支持通过比较 PyTorch 模型和 ONNX 模型的输出结果来进行验证。
要使用这个功能，要先通过 `pip install onnx onnxruntime` 安装所需的依赖包。

- 对于行为识别模型，请运行：

    ```shell
    python tools/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --shape $SHAPE --verify
    ```

- 对于时序检测模型，请运行：

    ```shell
    python tools/pytorch2onnx.py $CONFIG_PATH $CHECKPOINT_PATH --is-localizer --shape $SHAPE --verify
    ```

### 发布模型

`tools/publish_model.py` 脚本用于帮助用户发布模型。

在用户上传自己训练的模型到 MMAction2 的 AWS 服务器前，需要：

(1) 将模型的权重向量转化为 CPU 向量。
(2) 删除优化器状态信息。
(3) 计算模型检查点文件的哈希值，并将哈希值添加到文件名后。

```shell
python tools/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例如,

```shell
python tools/publish_model.py work_dirs/tsn_r50_1x1x3_100e_kinetics400_rgb/latest.pth tsn_r50_1x1x3_100e_kinetics400_rgb.pth
```

最终的输出文件名将会如 `tsn_r50_1x1x3_100e_kinetics400_rgb-{hash id}.pth`。

## 其他脚本

### 指标评价

`tools/analysis/eval_metric.py` 脚本会根据给定配置文件计算结果存储文件的某一评价指标值。

结果存储文件是通过在 `tools/test.py` 脚本中利用参数 `--out ${RESULT_FILE}` 指定的，其存储了整个模型的最终结果。

```shell
python tools/analysis/eval_metric.py ${CONFIG_FILE} ${RESULT_FILE} [--eval ${EVAL_METRICS}] [--cfg-options ${CFG_OPTIONS}] [--eval-options ${EVAL_OPTIONS}]
```

### 打印完整配置

`tools/analysis/print_config.py` 脚本会逐词打印整个配置文件的内容，并会扩展其所有的导入变量。

```shell
python tools/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
```
