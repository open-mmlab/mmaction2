# Tutorial 5: Use Multi-Grid From an Existed Training

## Prerequisites
1. First, your training should use step scheduler, since now multi-grid training doesn't support other scheduler type.
2. Second, this training method is only used for 3D models such as I3D, SlowFast that use dense-sampling strategy. Models such as TSN, TSM is not supported.

## Modify your config
A basic reference can be the config for multi-grid trained [I3D](../../configs/recognition/i3d/i3d_r50_multigrid_32x2x1_100e_kinetics400_rgb.py). You may follow the steps to use multi-grid from a standard stepwise [config](../../configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py).

1. Add multi grid config:
```python
...
multi_grid = dict(
    long_cycle=True,
    short_cycle=True,
    long_cycle_factors=((0.25, 0.5**0.5), (0.5, 0.5**0.5), (0.5, 1), (1, 1)),
    short_cycle_factors=(0.5, 0.5**0.5),
    epoch_factor=1.5,
    default_s=(224, 224))
...
```
For multi-grid training, it is claimed best to use both long-cycle and short-cycle. Our repo support long-cycle and long-plus-short-cycle, but not short alone. The factors all follow the default setting from [PySlowFast](https://github.com/facebookresearch/SlowFast). Note that the field `default_s` should be consist with the output scale of your data pipline. We only support square images as the original repo does, therefore we recommand you to use a `Resize` to resize the images into a square shape.

2. Add multi grid config in your train data config:
```python
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        # short_cycle_factors should be specific here
        short_cycle_factors=multi_grid['short_cycle_factors']),
        # default_s shuold be specific here
        default_s=multi_grid['default_s'],
```
Due to the convention of our codebase, which is that we build each module(model, data pipeline, runner etc.) seperately and the config variable is not global and writable, therefore you also need to add some information to data config:

3. Modify the train pipeline:
```python
# multi-grid use BatchSampler now, use CTHW
dict(type='FormatShape', input_format='CTHW'),
```
Short-cycle is implemented with a BatchSampler, which is not consist with other configs that use default Pytorch collate function, therefore we do not preserve an axis for 'batch', but leave the Sampler to collate samples at 'batch' dimension.

4. Press RUN:
Now the config is free to run!
