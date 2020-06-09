# Config System
We incorporate modular and inheritance (TODO) design into our config system, which is convenient to conduct various experiments.

## Config File Naming Style

We follow the style below to name config files. Contributors are advised to follow the same style.

```
{model}_[model setting]_{backbone}_[misc]_{data setting}_[gpu x batch_per_gpu]_{schedule}_{dataset}_{modality}
```

`{xxx}` is required field and `[yyy]` is optional.

- `{model}`: model type, e.g. `tin`, `i3d`, etc.
- `[model setting]`: specific setting for some models.
- `{backbone}`: backbone type, e.g. `r50` (ResNet-50), etc.
- `[misc]`: miscellaneous setting/plugins of model, e.g. `dense`, `2d`, `3d`, etc.
- `{data setting}`: frame sample setting in `{clip_len}x{frame_interval}x{num_clips}` format.
- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
- `{schedule}`: training schedule, e.g. `1x`, `2x`, `20e`, etc.
`1x` and `2x` means 12 epochs and 24 epochs respectively. `20e` is adopted in cascade models, which denotes 20 epochs.
- `{dataset}`: dataset name, e.g. `kinetics400`, `mmit`, etc.
- `{modality}`: frame modality, e.g. `rgb`, `flow`, etc.
