# 基准测试

这里将 MMAction2 与其他流行的代码框架和官方开源代码的速度性能进行对比

## 配置

### 硬件环境

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6146 CPU @ 3.20GHz

### 软件环境

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

### 评测指标

这里测量的时间是一轮训练迭代的平均时间，包括数据处理和模型训练。
训练速度以 s/iter 为单位，其值越低越好。注意，这里跳过了前 50 个迭代时间，因为它们可能包含设备的预热时间。

### 比较规则

这里以一轮训练迭代时间为基准，使用了相同的数据和模型设置对 MMAction2 和其他的视频理解工具箱进行比较。参与评测的其他代码库包括

- MMAction: commit id [7f3490d](https://github.com/open-mmlab/mmaction/tree/7f3490d3db6a67fe7b87bfef238b757403b670e3)(1/5/2020)
- Temporal-Shift-Module: commit id [8d53d6f](https://github.com/mit-han-lab/temporal-shift-module/tree/8d53d6fda40bea2f1b37a6095279c4b454d672bd)(5/5/2020)
- PySlowFast: commit id [8299c98](https://github.com/facebookresearch/SlowFast/tree/8299c9862f83a067fa7114ce98120ae1568a83ec)(7/7/2020)
- BSN(boundary sensitive network): commit id [f13707f](https://github.com/wzmsltw/BSN-boundary-sensitive-network/tree/f13707fbc362486e93178c39f9c4d398afe2cb2f)(12/12/2018)
- BMN(boundary matching network): commit id [45d0514](https://github.com/JJBOY/BMN-Boundary-Matching-Network/tree/45d05146822b85ca672b65f3d030509583d0135a)(17/10/2019)

为了公平比较，这里基于相同的硬件环境和数据进行对比实验。
使用的视频帧数据集是通过 [数据准备工具](/tools/data/kinetics/README.md) 生成的，
使用的视频数据集是通过 [该脚本](/tools/data/resize_video.py) 生成的，以快速解码为特点的，"短边 256，密集关键帧编码“的视频数据集。
正如以下表格所示，在对比正常的短边 256 视频时，可以观察到速度上的显著提升，尤其是在采样特别稀疏的情况下，如 [TSN](/configs/recognition/tsn/tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb.py)。

## 主要结果

### 行为识别器

| 模型  |输入| IO 后端 | 批大小 x GPU 数量 | MMAction2 (s/iter) | GPU 显存占用 (GB) | MMAction (s/iter)| GPU 显存占用 (GB) | Temporal-Shift-Module (s/iter) | GPU 显存占用 (GB) | PySlowFast (s/iter)| GPU 显存占用 (GB) |
| :--- | :---------------:|:---------------:| :---------------:| :---------------:  | :--------------------: | :----------------------------: | :-----------------: |:-----------------: |:-----------------: |:-----------------: |:-----------------: |
| [TSN](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py)| 256p rawframes |Memcached| 32x8|**[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_rawframes_memcahed_32x8.zip)** | 8.1 |[0.38](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/tsn_256p_rawframes_memcached_32x8.zip)|8.1| [0.42](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsn_256p_rawframes_memcached_32x8.zip)|10.5 | x |x |
| [TSN](/configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py)| 256p videos |Disk| 32x8|**[1.42](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_videos_disk_32x8.zip)** | 8.1 | x |x |x| x | TODO |TODO|
| [TSN](/configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py)| 256p dense-encoded video |Disk| 32x8|**[0.61](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsn_256p_fast_videos_disk_32x8.zip)**|8.1 | x |x| x |x| TODO |TODO|
|[I3D heavy](/configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py)|256p videos|Disk |8x8| **[0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_heavy_256p_videos_disk_8x8.zip)** |4.6|x |x| x |x| [0.44](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_i3d_r50_8x8_video.log) |4.6|
|[I3D heavy](/configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py)|256p dense-encoded video|Disk |8x8| **[0.35](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_heavy_256p_fast_videos_disk_8x8.zip)**| 4.6 | x | x | x | x | [0.36](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_i3d_r50_8x8_fast_video.log) |4.6|
| [I3D](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)|256p rawframes|Memcached|8x8| **[0.43](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/i3d_256p_rawframes_memcahed_8x8.zip)**|5.0 | [0.56](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction/i3d_256p_rawframes_memcached_8x8.zip)|5.0| x |x| x |x|
| [TSM](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py) |256p rawframes|Memcached| 8x8|**[0.31](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/tsm_256p_rawframes_memcahed_8x8.zip)** |6.9| x |x| [0.41](https://download.openmmlab.com/mmaction/benchmark/recognition/temporal_shift_module/tsm_256p_rawframes_memcached_8x8.zip) |9.1| x|x |
| [Slowonly](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.32](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowonly_256p_videos_disk_8x8.zip)** |3.1| TODO|TODO | x|x | [0.34](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowonly_r50_4x16_video.log)|3.4 |
| [Slowonly](/configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p dense-encoded video|Disk|8x8 | **[0.25](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowonly_256p_fast_videos_disk_8x8.zip)** |3.1| TODO |TODO| x |x| [0.28](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowonly_r50_4x16_fast_video.log) |3.4|
| [Slowfast](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p videos|Disk|8x8 | **[0.69](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowfast_256p_videos_disk_8x8.zip)**|6.1 | x |x| x |x| [1.04](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowfast_r50_4x16_video.log)|7.0 |
| [Slowfast](/configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py)|256p dense-encoded video|Disk|8x8 | **[0.68](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/slowfast_256p_fast_videos_disk_8x8.zip)** |6.1| x|x | x |x| [0.96](https://download.openmmlab.com/mmaction/benchmark/recognition/pyslowfast/pysf_slowfast_r50_4x16_fast_video.log)|7.0 |
| [R(2+1)D](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py)|256p videos |Disk| 8x8|**[0.45](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/r2plus1d_256p_videos_disk_8x8.zip)**|5.1 | x | x | x | x | x | x |
| [R(2+1)D](/configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py)|256p dense-encoded video |Disk| 8x8|**[0.44](https://download.openmmlab.com/mmaction/benchmark/recognition/mmaction2/r2plus1d_256p_fast_videos_disk_8x8.zip)** |5.1| x|x | x |x| x |x|

### 时序动作检测器

| Model | MMAction2 (s/iter) | BSN(boundary sensitive network) (s/iter) |BMN(boundary matching network) (s/iter)|
| :--- | :---------------: | :-------------------------------------: | :-------------------------------------: |
| BSN ([TEM + PEM + PGM](/configs/localization/bsn)) | **0.074(TEM)+0.040(PEM)** | 0.101(TEM)+0.040(PEM) | x |
| BMN ([bmn_400x100_2x8_9e_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py)) | **3.27** | x | 3.30 |

## 比较细节

### TSN

- **MMAction2**

```shell
# 处理视频帧
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_tsn configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py --work-dir work_dirs/benchmark_tsn_rawframes

# 处理视频
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_tsn configs/recognition/tsn/tsn_r50_video_1x1x3_100e_kinetics400_rgb.py --work-dir work_dirs/benchmark_tsn_video
```

- **MMAction**

```shell
python -u tools/train_recognizer.py configs/TSN/tsn_kinetics400_2d_rgb_r50_seg3_f1s1.py
```

- **Temporal-Shift-Module**

```shell
python main.py kinetics RGB --arch resnet50 --num_segments 3 --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 1 --batch-size 256 -j 32 --dropout 0.5 --consensus_type=avg --eval-freq=10 --npb --print-freq 1
```

### I3D

- **MMAction2**

```shell
# 处理视频帧
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_i3d configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py --work-dir work_dirs/benchmark_i3d_rawframes

# 处理视频
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_i3d configs/recognition/i3d/i3d_r50_video_heavy_8x8x1_100e_kinetics400_rgb.py --work-dir work_dirs/benchmark_i3d_video
```

- **MMAction**

```shell
python -u tools/train_recognizer.py configs/I3D_RGB/i3d_kinetics400_3d_rgb_r50_c3d_inflate3x1x1_seg1_f32s2.py
```

- **PySlowFast**

```shell
python tools/run_net.py   --cfg configs/Kinetics/I3D_8x8_R50.yaml   DATA.PATH_TO_DATA_DIR ${DATA_ROOT}   NUM_GPUS 8 TRAIN.BATCH_SIZE 64 TRAIN.AUTO_RESUME False LOG_PERIOD 1 SOLVER.MAX_EPOCH 1 > pysf_i3d_r50_8x8_video.log
```

可以通过编写一个简单的脚本对日志文件的 'time_diff' 域进行解析，以复现对应的结果。

### SlowFast

- **MMAction2**

```shell
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_slowfast configs/recognition/slowfast/slowfast_r50_video_4x16x1_256e_kinetics400_rgb.py --work-dir work_dirs/benchmark_slowfast_video
```

- **MMAction**

```shell
python tools/run_net.py   --cfg configs/Kinetics/SLOWFAST_4x16_R50.yaml   DATA.PATH_TO_DATA_DIR ${DATA_ROOT}   NUM_GPUS 8 TRAIN.BATCH_SIZE 64 TRAIN.AUTO_RESUME False LOG_PERIOD 1 SOLVER.MAX_EPOCH 1 > pysf_slowfast_r50_4x16_video.log
```

可以通过编写一个简单的脚本对日志文件的 'time_diff' 域进行解析，以复现对应的结果。

### SlowOnly

- **MMAction2**

```shell
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_slowonly configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py --work-dir work_dirs/benchmark_slowonly_video
```

- **PySlowFast**

```shell
python tools/run_net.py   --cfg configs/Kinetics/SLOW_4x16_R50.yaml   DATA.PATH_TO_DATA_DIR ${DATA_ROOT}   NUM_GPUS 8 TRAIN.BATCH_SIZE 64 TRAIN.AUTO_RESUME False LOG_PERIOD 1 SOLVER.MAX_EPOCH 1 > pysf_slowonly_r50_4x16_video.log
```

可以通过编写一个简单的脚本对日志文件的 'time_diff' 域进行解析，以复现对应的结果。

### R2plus1D

- **MMAction2**

```shell
bash tools/slurm_train.sh ${PARTATION_NAME} benchmark_r2plus1d configs/recognition/r2plus1d/r2plus1d_r34_video_8x8x1_180e_kinetics400_rgb.py --work-dir work_dirs/benchmark_r2plus1d_video
```
