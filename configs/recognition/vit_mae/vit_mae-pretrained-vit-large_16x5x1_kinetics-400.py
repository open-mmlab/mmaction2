_base_ = ['vit_mae-pretrained-vit-base_16x5x1_kinetics-400.py']

# model settings
model = dict(backbone=dict(embed_dim=1024, depth=24, num_heads=16))
