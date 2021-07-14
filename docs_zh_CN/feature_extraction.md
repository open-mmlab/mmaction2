# 特征提取

MMAction2 为特征提取提供了便捷使用的脚本。

## 片段级特征提取

片段级特征提取是从长度一般为几秒到几十秒不等的剪辑片段中提取深度特征。从每个片段中提取的特征是一个 n 维向量。当进行多视图特征提取时，例如 n 个片段 × m 种裁剪，提取的特征将会是 n*m 个视图的平均值。

在应用片段级特征提取之前，用户需要准备一个视频列表包含所有想要进行特征提取的视频。例如，由 UCF101 中视频组成的视频列表如下：

```
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02.avi
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03.avi
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c04.avi
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c05.avi
...
YoYo/v_YoYo_g25_c01.avi
YoYo/v_YoYo_g25_c02.avi
YoYo/v_YoYo_g25_c03.avi
YoYo/v_YoYo_g25_c04.avi
YoYo/v_YoYo_g25_c05.avi
```

假设 UCF101 中的视频所在目录为 `data/ucf101/videos`，视频列表的文件名为 `ucf101.txt`，使用 TSN（Kinetics-400 预训练）从 UCF101 中提取片段级特征，用户可以使用脚本如下：

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/tsn/tsn_r50_clip_feature_extraction_1x1x3_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

被提取的特征存储于 `ucf101_feature.pkl`。

用户也可以使用分布式片段级特征提取。以下是使用拥有 8 gpus 的计算节点的示例。

```shell
bash tools/misc/dist_clip_feature_extraction.sh \
configs/recognition/tsn/tsn_r50_clip_feature_extraction_1x1x3_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth \
8 \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

使用 SlowOnly（Kinetics-400 预训练）从 UCF101 中提取片段级特征，用户可以使用脚本如下：

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/slowonly/slowonly_r50_clip_feature_extraction_4x16x1_rgb.py \
https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

这两个配置文件展示了用于特征提取的最小配置。用户也可以使用其他存在的配置文件进行特征提取，只要注意使用视频数据进行训练和测试，而不是原始帧数据。

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```
