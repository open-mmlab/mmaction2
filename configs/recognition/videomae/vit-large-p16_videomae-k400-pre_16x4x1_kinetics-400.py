_base_ = ['vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py']

# model settings
model = dict(
    backbone=dict(embed_dims=1024, depth=24, num_heads=16),
    cls_head=dict(in_channels=1024))
