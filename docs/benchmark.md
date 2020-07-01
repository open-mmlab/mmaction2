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

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| TSN        | Kinetics400 | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | HAN           |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
|            | Sthv1       | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | HAN           |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
| TSM        |             | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | HAN           |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
| TIN        |             | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | HAN           |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
| Slowfast   | Kinetics400 | 4x16x1@8(video) | mmaction-lite |0.80|6203|ready|
|            | Kinetics400 | 4x16x1@8(video) | PySlowfast    |1.40|6850|ready|
|            | Kinetics400 | 8x8x1@8(video) | mmaction-lite |1.05|9062|ready|
|            | Kinetics400 | 8x8x1@8(video) | PySlowfast |1.41|10230|ready|
| Slowonly   | Kinetics400 | 4x16x1@8(video) | mmaction-lite |0.30|3158|ready|
|            | Kinetics400 | 4x16x1@8(video)| PySlowfast    |1.03|3481|ready|
|            | Kinetics400 | 8x8x1@8(video) | mmaction-lite |0.50|5820|ready|
|            | Kinetics400 | 8x8x1@8(video)| PySlowfast    |1.29|6400|ready|
| R(2+1)D    | Kinetics400 | 8x8x1@6(frame) | mmaction-lite |0.48|3998|ready|
|            | Kinetics400 | 32x2x1@6(frame) | mmaction-lite |1.2855|12974|ready|
