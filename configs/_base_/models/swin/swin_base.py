_base_ = 'swin_tiny.py'
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.3),
    cls_head=dict(in_channels=1024))
