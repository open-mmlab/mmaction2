_base_ = [
    'slowfast_kinetics400_pretrained_r50_8x8x1_cosine_10e_8xb6_ava22_rgb.py'
]

model = dict(roi_head=dict(bbox_roi_extractor=dict(temporal_pool_mode='max')))

load_from = ('https://download.openmmlab.com/mmaction/recognition/slowfast/'
             'slowfast_r50_8x8x1_256e_kinetics400_rgb/'
             'slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth')
