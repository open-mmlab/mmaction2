_base_ = [
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        bottleneck_mode='ip',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ipcsn_from_scratch_r152_ig65m_20210617-c4b99d38.pth'  # noqa: E501
    ))
