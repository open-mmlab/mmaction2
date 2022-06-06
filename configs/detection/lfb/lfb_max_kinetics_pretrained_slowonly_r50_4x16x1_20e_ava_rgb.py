_base_ = './lfb_max_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py',

model = dict(roi_head=dict(shared_head=dict(fbo_cfg=dict(type='max'))))