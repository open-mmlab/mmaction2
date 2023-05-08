_base_ = ['vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400.py']

# model settings
model = dict(
    backbone=dict(embed_dims=384, depth=12, num_heads=6),
    cls_head=dict(in_channels=384))
