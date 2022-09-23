_base_ = ['videomae-pretrained-vit-base_16x4x1_kinetics-400.py']

# model settings
model = dict(
    backbone=dict(embed_dims=1024, depth=24, num_heads=16),
    cls_head=dict(in_channels=1024))
