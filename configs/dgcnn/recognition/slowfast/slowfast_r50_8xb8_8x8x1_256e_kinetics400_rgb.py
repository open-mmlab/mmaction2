from mmengine.config import read_base

with read_base():
    from .slowfast_r50_8xb8_4x16x1_256e_kinetics400_rgb import *

model = dict(
    backbone=dict(
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(fusion_kernel=7)))
