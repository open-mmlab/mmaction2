# Benchmark

We compare our results with some other popular frameworks in terms of speed and performance.

### Hardware

- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

### Software environment

- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | ckpt & log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| TSN        | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | mmaction-lite |0.2966(0.0030)|8339|            |
|            | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | open-mmaction |0.3659(0.0102)|8245|            |
| I3D        | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| mmaction-lite |0.4528(0.1458)|5169|            |
|            | Kinetics400 | i3d_r50_fast32x2x1_100e_kinetics400_rgb| mmaction-lite |0.3886(0.3733)|5169|            |
|            | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| open-mmaction |0.3659(0.0102)|5065|            |
| TSM        | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | mmaction-lite |0.3052(0.0095)|7077|            |
|            | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | HAN           |0.3843(0.0143)|9337|            |
| Slowfast   | Kinetics400 | 4x16x1(video) | mmaction-lite |           |        |            |
|            | Kinetics400 | 4x16x1(video) | PySlowfast    |           |        |            |
|            | Kinetics400 | 8x8x1(video) | mmaction-lite |           |        |            |
| Slowonly   | Kinetics400 | 4x16x1(video) | mmaction-lite |           |        |            |
|            | Kinetics400 | 4x16x1(video)| PySlowfast    |           |        |            |
|            | Kinetics400 | 8x8x1(video) | mmaction-lite |           |        |            |
| R(2+1)D    | Kinetics400 | 32x2x1(frame) | mmaction-lite |           |        |            |
|            | Kinetics400 | 8x8x1(frame) | mmaction-lite |           |        |            |
