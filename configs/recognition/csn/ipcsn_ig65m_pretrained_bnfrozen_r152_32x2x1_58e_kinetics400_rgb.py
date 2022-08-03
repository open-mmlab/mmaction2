_base_ = [
    'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_8xb12_kinetics400_rgb.py'
]

# model settings
model = dict(
    backbone=dict(
        norm_eval=True, bn_frozen=True, bottleneck_mode='ip', pretrained=None))
