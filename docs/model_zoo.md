# Model Zoo

## Mirror Sites

We use aliyun as the main site to host our model zoo. The model urls start with `https://openmmlab.oss-accelerate.aliyuncs.com`.

## Commn settings

- We use distributed training.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.

## Recognition

### TSN

Please refer to [TSN](/configs/recognition/tsn) for details.

### TSM

Please refer to [TSM](/configs/recognition/tsm) for details.

### R(2+1)D

Please refer to [R(2+1)D](/configs/recognition/r2plus1d) for details.

### I3D

Please refer to [I3D](/configs/recognition/i3d) for details.

### SlowOnly

Please refer to [SlowOnly](/configs/recognition/slowonly) for details.

### SlowFast

Please refer to [SlowFast](/configs/recognition/slowfast) for details.

## Localization

### BMN

Please refer to [BMN](/configs/localization/bmn) for details.

### BSN

Please refer to [BSN](/configs/localization/bsn) for details.
