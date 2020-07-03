# Benchmark

We compare our results with some other popular frameworks in terms of speed and performance.

## Results

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| TSN        | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | mmaction-lite |0.2966|8339|ready|
|            | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb(video) | mmaction-lite |0.4165|8339|ready|
|            | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | open-mmaction |0.3659|8245|ready|
| I3D        | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| mmaction-lite |0.4528|5169|ready|
|            | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb(video)| mmaction-lite |0.4795|5169|ready|
|            | Kinetics400 | i3d_r50_fast32x2x1_100e_kinetics400_rgb| mmaction-lite |0.3886|5169|ready
|            | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| open-mmaction |0.5873|5065|ready|
| TSM        | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | mmaction-lite |0.3052|7077|ready|
|            | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb(video) | mmaction-lite |0.3027|7077|ready|
|            | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | HAN           |0.3843|9337|ready|
| Slowfast   | Kinetics400 | slowfast_r50_4x16x1_256e_kinetics400_rgb(video) | mmaction-lite |0.80|6203|ready|
|            | Kinetics400 | slowfast_r50_4x16x1_256e_kinetics400_rgb(video) | PySlowfast    |1.40|6850|ready|
|            | Kinetics400 | slowfast_r50_8x8x1_256e_kinetics400_rgb(video) | mmaction-lite |1.05|9062|ready|
|            | Kinetics400 | slowfast_r50_8x8x1_256e_kinetics400_rgb(video) | PySlowfast |1.41|10230|ready|
| Slowonly   | Kinetics400 | slowonly_r50_4x16x1_256e_kinetics400_rgb(video) | mmaction-lite |0.30|3158|ready|
|            | Kinetics400 | slowonly_r50_4x16x1_256e_kinetics400_rgb(video)| PySlowfast    |1.03|3481|ready|
|            | Kinetics400 | slowonly_r50_8x8x1_256e_kinetics400_rgb(video) | mmaction-lite |0.50|5820|ready|
|            | Kinetics400 | slowonly_r50_8x8x1_256e_kinetics400_rgb(video)| PySlowfast    |1.29|6400|ready|
| R(2+1)D    | Kinetics400 | r2plus1d_r34_8x8x1_180e_kinetics400_rgb(frame) | mmaction-lite |0.48|3998|ready|
|            | Kinetics400 | r2plus1d_r34_8x8x1_180e_kinetics400_rgb(frame) | mmaction-lite |1.29|12974|ready|
|            | Kinetics400 | r2plus1d_r34_8x8x1_180e_kinetics400_rgb(video) | mmaction-lite |0.8340(0.0725)|10339|    |


| Model      | MMAction    |  [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-modulen)     |
| ---------- | :---------: | :---------:|
| TSM        | 0.30      | 0.38 |


| Model      | MMAction    | [PySlowFast](https://github.com/facebookresearch/SlowFast) |
| ---------- | :---------: | :----------: |
| Slowfast(4x16x1)| 0.80 | 1.40 |
| Slowfast(8x8x1) | 1.05 | 1.41 |
| Slowonly(4x16x1)| 0.30 | 1.03 |
| Slowonly(8x8x1) | 0.50 | 1.29 |
| R(2+1)D         | 0.48 | 1.29 |

### Localizers

| Model | MMAction     | [BSN-boundary-sensitive-network](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch)  |
| :----------------------------: | :-----------: | :--------: |
| BSN(400x100,1x16) | 0.074(TEM)+0.040(PEM) | 0.101(TEM)+0.040(PEM) |
| BSN(400x100,2x8) | 3.27 | 3.30 |
