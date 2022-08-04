_base_ = [
    'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=True,
        bn_frozen=True,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'))
