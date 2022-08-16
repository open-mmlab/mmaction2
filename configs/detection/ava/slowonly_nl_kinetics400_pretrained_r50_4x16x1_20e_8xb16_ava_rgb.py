_base_ = ['slowonly_kinetics400_pretrained_r50_4x16x1_20e_8xb16_ava_rgb.py']

model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN3d', requires_grad=True),
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=True,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')))

load_from = (
    'https://download.openmmlab.com/mmaction/recognition/slowonly/'
    'slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/'
    'slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb_20210308-0d6e5a69.pth'  # noqa: E501
)
