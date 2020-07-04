# Benchmark

We compare our results with some other popular frameworks in terms of speed.

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

### MMAction vs [MMAction(old-version)](https://github.com/open-mmlab/mmaction)

| architecture | depth | clip len x frame interval x num clips | MMAction(s/iter) | MMAction(old)(s/iter) | config |
| :----------: | :---: | :-----------------------------------: | :---------: | :--------------: | :----: |
| TSN | R50 | 1 x 1 x 3 | 0.29 | 0.36 | [tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py) |
| I3D | R50 | 32 x 2 x 1| 0.45 | 0.58 | [i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py) |

### MMAction vs [temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-modulen)

| architecture| depth | clip len x frame interval x num clips | MMAction(s) | temporal-shift-module(s) | config |
| :---------: | :---: | :-----------------------------------: | :---------: | :----------------------: | :----: |
| TSM | R50 | 1 x 1 x 8 | 0.30 | 0.38 | [tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py)

### MMAction vs [PySlowFast](https://github.com/facebookresearch/SlowFast)

| architecture| depth | clip len x frame interval x num clips | MMAction(s) | PySlowFast(s) | config |
| :---------: | :---: | :-----------------------------------: | :---------: | :-----------: | :----: |
| Slowfast| R50 | 4 x 16 x 1| 0.80 | 1.40 | [slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py)
| Slowfast| R50 | 8 x 8 x 1 | 1.05 | 1.41 | [slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py)
| Slowonly| R50 | 4 x 16 x1 | 0.30 | 1.03 | [slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)
| Slowonly| R50 | 8 x 8 x 1 | 0.50 | 1.29 | [slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py)
| R(2+1)D | R34 | 8 x 8 x 1 | 0.48 | 1.29 | [r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py)

## Localizers

### MMAction vs [BSN-boundary-sensitive-network](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch)

| architecture | MMAction(s) | BSN-boundary-sensitive-network(s) | config |
| :----------: | :---------: | :-------------------------------: | :----: |
| BSN | 0.074(TEM)+0.040(PEM) | 0.101(TEM)+0.040(PEM) | [TEM + PEM + PGM](/configs/localization/bsn)
| BMN | 3.27 | 3.30 | [BMN](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py)



| Model | MMAction(s/iter) | MMAction(old)(s/iter) | temporal-shift-module(s/iter) | PySlowFast |
| :---: | :--------------: | :-------------------: | :---------------------------: | :--------: |
| TSN ([tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py)) | 0.29 | 0.36 | None | None |
| I3D ([i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)) | 0.45 | 0.58 | None | None |
|
