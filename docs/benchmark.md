# Benchmark

We compare our results with some other popular frameworks in terms of speed and performance.

## Results

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| TSN        | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | mmaction-lite |0.2966(0.0030)|8339|            |
|            | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb(video) | mmaction-lite |0.4165(0.0098)|8339|            |
|            | Kinetics400 | tsn_r50_1x1x3_100e_kinetics400_rgb | open-mmaction |0.3659(0.0102)|8245|            |
| I3D        | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| mmaction-lite |0.4528(0.1458)|5169|            |
|            | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb(video)| mmaction-lite |0.4795(0.0032)|5169|            |
|            | Kinetics400 | i3d_r50_fast32x2x1_100e_kinetics400_rgb| mmaction-lite |0.3886(0.3733)|5169|            |
|            | Kinetics400 | i3d_r50_32x2x1_100e_kinetics400_rgb| open-mmaction |0.5873(0.0867)|5065|            |
| TSM        | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | mmaction-lite |0.3052(0.0095)|7077|            |
|            | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb(video) | mmaction-lite |0.3027(0.0089)|7077|            |
|            | Kinetics400 | tsm_r50_1x1x8_100e_kinetics400_rgb | HAN           |0.3843(0.0143)|9337|            |
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


### Localizers

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | ckpt & log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| BSN       | ActivityNet | bsn_400x100_1x16_20e_activitynet_feature | mmaction-lite |0.074(TEM)+0.040(PEM)|41(TEM)+25(PEM)|            |
| BSN       | ActivityNet | bsn_400x100_1x16_20e_activitynet_feature | [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch) |0.101(TEM)+0.040(PEM)|54(TEM)+34(PEM)|            |
| BMN       | ActivityNet | bsn_400x100_2x8_9e_activitynet_feature | mmaction-lite |3.27|5420|            |
| BMN       | ActivityNet | bsn_400x100_2x8_9e_activitynet_feature | [repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network) |3.30|5780|            |
