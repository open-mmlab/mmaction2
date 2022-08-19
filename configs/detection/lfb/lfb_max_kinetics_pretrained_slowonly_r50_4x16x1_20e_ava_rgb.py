_base_ = ['lfb_nl_kinetics_pretrained_slowonly_r50_4x16x1_20e_ava_rgb.py']

model = dict(
    roi_head=dict(
        shared_head=dict(fbo_cfg=dict(type='max')),
        bbox_head=dict(in_channels=4096)))
