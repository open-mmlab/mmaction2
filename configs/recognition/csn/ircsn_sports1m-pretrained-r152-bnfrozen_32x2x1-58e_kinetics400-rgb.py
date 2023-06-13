_base_ = [
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True,
        bn_frozen=True,
        bottleneck_mode='ir',
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_sports1m_20210617-bcc9c0dd.pth'  # noqa: E501
    ),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[110.2008, 100.63983, 95.99475],
        std=[58.14765, 56.46975, 55.332195],
        format_shape='NCTHW'))
