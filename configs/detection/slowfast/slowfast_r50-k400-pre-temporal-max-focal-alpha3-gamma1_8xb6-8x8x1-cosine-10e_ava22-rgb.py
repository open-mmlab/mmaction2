_base_ = [
    'slowfast_kinetics400-pretrained-r50_8xb6-8x8x1-cosine-10e_ava22-rgb.py'
]

model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(temporal_pool_mode='max'),
        bbox_head=dict(focal_alpha=3.0, focal_gamma=1.0)))
