_base_ = 'swin_tiny.py'
model = dict(
    backbone=dict(
        depths=[2, 2, 18, 2],
        embed_dim=192,
        num_heads=[6, 12, 24, 48],
        drop_path_rate=0.4),
    cls_head=dict(in_channels=1536))
