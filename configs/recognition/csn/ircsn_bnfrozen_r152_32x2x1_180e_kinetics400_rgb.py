_base_ = [
    'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True, bn_frozen=True, bottleneck_mode='ir', pretrained=None),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[110.2008, 100.63983, 95.99475],
        std=[58.14765, 56.46975, 55.332195],
        format_shape='NCTHW'))
