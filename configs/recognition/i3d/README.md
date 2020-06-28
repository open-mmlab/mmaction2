# I3D

## Model Zoo

### Kinetics-400

|config | pretrain | top1 acc| top5 acc | gpu_mem(M) | iter time(s) | ckpt | log|
|-|-|-|-|-|-|-|-|
|[i3d_r34_32x2x1_100e_kinetics400_rgb](i3d_r34_32x2x1_100e_kinetics400_rgb.py) | ResNet34 |68.37|88.15| x | x | [ckpt]() | [log]()|
|[i3d_r50_32x2x1_100e_kinetics400_rgb](i3d_r50_32x2x1_100e_kinetics400_rgb.py) | ResNet50 |72.68|90.78| x | x | [ckpt]() | [log]()|
|[i3d_r50_dense_32x2x1_100e_kinetics400_rgb](i3d_r50_dense_32x2x1_100e_kinetics400_rgb.py) | ResNet50 |72.77|90.57| x | x | [ckpt]() | [log]()|
|[i3d_r50_fast_32x2x1_100e_kinetics400_rgb](i3d_r50_fast_32x2x1_100e_kinetics400_rgb.py) | ResNet50 |72.32|90.72| x | x | [ckpt]() | [log]()|
|[i3d_r50_video_3d_32x2x1_100e_kinetics400_rgb](i3d_r50_video_32x2x1_100e_kinetics400_rgb.py) | ResNet50 | x | x | x | x | [ckpt]() | [log]()|

## Data

1. Make a dataset folder under the path `$MMACTION/data`.
2. Put the data sub folders (commonly including `rawframes_train/` + `rawframes_val/` or `video_train` + `video_val`) under `$MMACTION/data/dataset_name`.
It is recommended to symlink the dataset root to the corresponding folders.
3. Put the annotation (commonly including `ann_file_train.txt` + `ann_file_val.txt`) files under `$MMACTION/data/dataset_path` under `$MMACTION/data/dataset_name`.
4. Finally, make sure your folder structure same with the tree structure below.
If your folder structure is different, you can also change the corresponding paths in config files.
```
mmaction
├── mmaction
├── tools
├── config
├── data
│   ├── kinetics400
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── kinetics_train_list.txt
│   │   ├── kinetics_val_list.txt
│   ├── ucf101
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── ucf101_train_list.txt
│   │   ├── ucf101_val_list.txt
│   ├── sth-v1
│   │   ├── rawframes_train
│   │   ├── rawframes_val
│   │   ├── sth-v1_train_list.txt
│   │   ├── sth-v1_val_list.txt
...
```

## Checkpoint
Put the checkpoint required under `$MMACTION/checkpoints`. The checkpoints can be found at [here]().

## Train
You can use the following command to train a model.
```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

Example: train I3D model on Kinetics-400 dataset in a deterministic option with periodic validation.
```shell
python tools/train.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    --work_dir work_dirs/i3d_r50_32x2x1_100e_kinetics400_rgb \
    --validate --seed 0 --deterministic
```

For more details, you can refer to **Training setting** part in [getting_started](../../../docs/getting_started.md).

## Test
You can use the following command to test a model.
```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [optional arguments]
```

Example: test I3D model on Kinetics-400 dataset and dump the result to a json file.
```shell
python tools/test.py configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py \
    checkpoints/SOME_CHECKPOINT.pth --eval top_k_accuracy mean_class_accuracy \
    --out result.json
```

For more details, you can refer to **Test a dataset** part in [getting_started](../../../docs/getting_started.md).
