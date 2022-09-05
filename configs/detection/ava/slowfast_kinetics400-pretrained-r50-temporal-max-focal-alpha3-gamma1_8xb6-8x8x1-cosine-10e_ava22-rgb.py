_base_ = [
    'slowfast_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb.py'
]

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(temporal_pool_mode='max'),
        bbox_head=dict(focal_alpha=3.0, focal_gamma=1.0)))