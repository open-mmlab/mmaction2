# Feature Extraction

We provide easy to use scripts for feature extraction.

## Clip-level Feature Extraction

Clip-level feature extraction extract deep feature from a video clip, which usually lasts several to tens of seconds. The extracted feature is an n-dim vector for each clip. When performing multi-view feature extraction, e.g. n clips x m crops, the extracted feature will be the average of the n * m views.

Before applying clip-level feature extraction, you need to prepare a video list (which include all videos that you want to extract feature from). For example, the video list for videos in UCF101 will look like:

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

Assume the root of UCF101 videos is `data/ucf101/videos` and the name of the video list is `ucf101.txt`, to extract clip-level feature of UCF101 videos with Kinetics-400 pretrained TSN, you can use the following script:

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/tsn/tsn_r50_clip_feature_extraction_1x1x3_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

and the extracted feature will be stored in `ucf101_feature.pkl`

You can also use distributed clip-level feature extraction. Below is an example for a node with 8 gpus.

```shell
bash tools/misc/dist_clip_feature_extraction.sh \
configs/recognition/tsn/tsn_r50_clip_feature_extraction_1x1x3_rgb.py \
https://download.openmmlab.com/mmaction/recognition/tsn/tsn_r50_320p_1x1x3_100e_kinetics400_rgb/tsn_r50_320p_1x1x3_100e_kinetics400_rgb_20200702-cc665e2a.pth \
8 \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

To extract clip-level feature of UCF101 videos with Kinetics-400 pretrained SlowOnly, you can use the following script:

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/slowonly/slowonly_r50_clip_feature_extraction_4x16x1_rgb.py \
https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```

The two config files demonstrates what a minimal config file for feature extraction looks like. You can also use other existing config files for feature extraction, as long as they use videos rather than raw frames for training and testing:

```shell
python tools/misc/clip_feature_extraction.py \
configs/recognition/slowonly/slowonly_r50_video_4x16x1_256e_kinetics400_rgb.py \
https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb/slowonly_r50_video_320p_4x16x1_256e_kinetics400_rgb_20201014-c9cdc656.pth \
--video-list ucf101.txt \
--video-root data/ucf101/videos \
--out ucf101_feature.pkl
```
