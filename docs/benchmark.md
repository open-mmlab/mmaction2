# Benchmark

We compare our results with some other popular frameworks in terms of speed and performance.

## Results

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | log |
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


### Localizers

| Model      | Dataset     | Setting  | Framework     | Iter time | Memory | ckpt & log |
| ---------- | ----------- | -------- | ------------- | --------- | ------ | ---------- |
| BSN       | ActivityNet | bsn_400x100_1x16_20e_activitynet_feature | mmaction-lite |0.074(TEM)+0.040(PEM)|41(TEM)+25(PEM)|            |
| BSN       | ActivityNet | bsn_400x100_1x16_20e_activitynet_feature | [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch) |0.101(TEM)+0.040(PEM)|54(TEM)+34(PEM)|            |
| BMN       | ActivityNet | bsn_400x100_2x8_9e_activitynet_feature | mmaction-lite |3.27|5420|            |
| BMN       | ActivityNet | bsn_400x100_2x8_9e_activitynet_feature | [repo](https://github.com/JJBOY/BMN-Boundary-Matching-Network) |3.30|5780|            |
