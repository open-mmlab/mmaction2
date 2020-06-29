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
| PySlowfast |             | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | PySlowfast    |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
| R(2+1)D    |             | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | PySlowfast    |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
| R(2+1)D    |             | Setting1 | mmaction-lite |           |        |            |
|            |             |          | open-mmaction |           |        |            |
|            |             |          | PySlowfast    |           |        |            |
|            |             | Setting2 | mmaction-lite |           |        |            |
