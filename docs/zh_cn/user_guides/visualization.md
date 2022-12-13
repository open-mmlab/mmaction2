# 可视化工具

## 对数据集可视化

你可以使用`tools/analysis_tools/browse_dataset.py`去可视化数据集。

```shell
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [ARGS]
```

| 参数                            | 含义                                                                                                                                                                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                   | 配置文件的路径。                                                                                                                                                          |
| `--output-dir OUTPUT_DIR`       | 如果没有display显示接口，你能将可视化结果保存到`OUTPUT_DIR`，默认为None。                                                                                                 |
| `--show-frames`                 | 如果你拥有显示接口，会显示视频的帧内容，默认为False。                                                                                                                     |
| `--phase PHASE`                 | 想要可视化的数据集阶段，接受`train`, `test` 和`val`. 默认为`train`。                                                                                                      |
| `--show-number SHOW_NUMBER`     | 选择可视化的图像数量，必须比0大，如果数量比数据集长度更大，则展示数据集中的所有图像，默认为"sys.maxsize"，展示数据集中所有图像。                                          |
| `--show-interval SHOW_INTERVAL` | 显示的间隔，默认为2The interval of show (s). Defaults to 2。                                                                                                              |
| `--mode MODE`                   | 显示模式：显示原始视频或者变换后的视频。`original` 表示显示从硬盘中导入的视频，而`transformed` 表示显示变换后的视频，默认为`transformed`。                                |
| `--cfg-options CFG_OPTIONS`     | 覆盖一些正在使用的config配置的设置，像”xxx=yyy“形式的键值对将会被合并进config配置文件。如果将被覆盖的是一个列表，它的形式将是`key="[a,b]"` 或 `key=a,b`的格式。该参数还允许嵌套列表/元组值，例如`key="[(a,b),(c,d)]"`. 请注意，引号是必需的，不允许有空格。 |
