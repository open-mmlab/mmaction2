_base_ = ['slowfast_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(with_global=True),
        bbox_head=dict(in_channels=4608)))
