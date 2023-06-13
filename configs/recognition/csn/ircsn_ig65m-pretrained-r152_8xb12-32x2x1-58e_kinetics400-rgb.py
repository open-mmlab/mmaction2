_base_ = [
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
]

model = dict(
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/recognition/csn/'
        'ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth'))
