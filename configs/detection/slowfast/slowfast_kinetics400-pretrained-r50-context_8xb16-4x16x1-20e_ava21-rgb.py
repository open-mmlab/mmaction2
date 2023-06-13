_base_ = ['slowfast_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py']

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(with_global=True),
        bbox_head=dict(in_channels=4608)))
