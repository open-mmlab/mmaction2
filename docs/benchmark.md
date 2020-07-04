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

| Model | MMAction (s/iter) | MMAction V0.1 (s/iter) | temporal-shift-module (s/iter) | PySlowFast (s/iter) |
| :---: | :---------------: | :--------------------: | :----------------------------: | :-----------------: |
| TSN ([tsn_r50_1x1x3_100e_kinetics400_rgb](/configs/recognition/tsn/tsn_r50_1x1x3_100e_kinetics400_rgb.py))   | 0.29 | 0.36 | 0.45 | None |
| I3D ([i3d_r50_32x2x1_100e_kinetics400_rgb](/configs/recognition/i3d/i3d_r50_32x2x1_100e_kinetics400_rgb.py)) | 0.45 | 0.58 | None | None |
| TSM ([tsm_r50_1x1x8_50e_kinetics400_rgb](/configs/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb.py))     | 0.30 | None | 0.38 | None |
| Slowonly ([slowonly_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_4x16x1_256e_kinetics400_rgb.py)) | 0.30 | None | None | 1.03 |
| Slowonly ([slowonly_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_rgb.py))   | 0.50 | None | None | 1.29 |
| Slowfast ([slowfast_r50_4x16x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_4x16x1_256e_kinetics400_rgb.py)) | 0.80 | None | None | 1.40 |
| Slowfast ([slowfast_r50_8x8x1_256e_kinetics400_rgb](/configs/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb.py))   | 1.05 | None | None | 1.41 |
| R(2+1)D ([r2plus1d_r34_8x8x1_180e_kinetics400_rgb](/configs/recognition/r2plus1d/r2plus1d_r34_8x8x1_180e_kinetics400_rgb.py))    | 0.48 | None | None | None |

## Localizers

| Model | MMAction (s/iter) | BSN-boundary-sensitive-network (s/iter) |
| :---: | :---------------: | :-------------------------------------: |
| BSN ([TEM + PEM + PGM](/configs/localization/bsn)) | 0.074(TEM)+0.040(PEM) | 0.101(TEM)+0.040(PEM) |
| BMN ([bmn_400x100_2x8_9e_activitynet_feature](/configs/localization/bmn/bmn_400x100_2x8_9e_activitynet_feature.py)) | 3.27 | 3.30 |
