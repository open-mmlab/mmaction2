_base_ = ['vit_mae-pretrained-vit-b_16x5x1_kinetics-400.py']

# model settings
model = dict(
    type='Recognizer3D', backbone=dict(embed_dim=1024, depth=24, num_heads=16))
