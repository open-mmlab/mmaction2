from mmengine.config import read_base

with read_base():
    from .slowfast_r50_8xb8_8x8x1_256e_kinetics400_rgb import *

model = dict(
    backbone=dict(slow_pathway=dict(depth=101), fast_pathway=dict(depth=101)))
