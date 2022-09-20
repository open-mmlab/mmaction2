_base_ = "swin_tiny.py"
model = dict(backbone=dict(depths=[2, 2, 18, 2], drop_path_rate=0.2))
