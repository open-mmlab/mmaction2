from mmengine.config import read_base

with read_base():
    from .slowfast_r50_8xb8_4x16x1_256e_kinetics400_rgb import *

model = dict(backbone=dict(slow_pathway=dict(depth=101)))

optim_wrapper = dict(optimizer=dict(lr=0.1 * 4))
