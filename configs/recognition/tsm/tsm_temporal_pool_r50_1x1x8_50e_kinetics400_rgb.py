_base_ = ['./tsm_r50_1x1x8_50e_kinetics400_rgb.py']

# model settings
model = dict(
    backbone=dict(temporal_pool=True), cls_head=dict(temporal_pool=True))

# runtime settings
work_dir = './work_dirs/tsm_temporal_pool_r50_1x1x8_100e_kinetics400_rgb/'
