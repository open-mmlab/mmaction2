_base_ = [
    'ircsn_ig65m-pretrained-r152-bnfrozen_8xb12-32x2x1-58e_kinetics400-rgb.py'
]

# model settings
model = dict(
    backbone=dict(bottleneck_mode='ir', pretrained=None),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[110.2008, 100.63983, 95.99475],
        std=[58.14765, 56.46975, 55.332195],
        format_shape='NCTHW'))
