_base_ = [
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=True,
        bn_frozen=True,
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth'))
