# Visualization Tools

## Visualize dataset

You can use `tools/analysis_tools/browse_dataset.py` to visualize video datasets:

```bash
python tools/analysis_tools/browse_dataset.py ${CONFIG_FILE} [ARGS]
```

| ARGS                            | Description                                                                                                                                                               |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONFIG_FILE`                   | The path to the config file.                                                                                                                                              |
| `--output-dir OUTPUT_DIR`       | If there is no display interface, you can save the visualization results to `OUTPUT_DIR`. Defaults to None                                                                |
| `--show-frames`                 | Display the frames of the video if you have the display interface. Defaults to False.                                                                                     |
| `--phase PHASE`                 | Phase of the dataset to visualize, accept `train`, `test` and `val`. Defaults to `train`.                                                                                 |
| `--show-number SHOW_NUMBER`     | Number of images selected to visualize, must bigger than 0. Jf the number is bigger than length of dataset, show all the images in dataset. Defaults to "sys.maxsize", show all images in dataset |
| `--show-interval SHOW_INTERVAL` | The interval of show (s). Defaults to 2.                                                                                                                                  |
| `--mode MODE`                   | Display mode: display original videos or transformed videos. `original` means show videos load from disk while `transformed` means to show videos after transformed. Defaults to `transformed`. |
| `--cfg-options CFG_OPTIONS`     | Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either `key="[a,b]"` or `key=a,b`. The argument also allows nested list/tuple values, e.g. `key="[(a,b),(c,d)]"`. Note that the quotation marks are necessary and that no white space is allowed. |
