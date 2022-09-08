_base_ = ['slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb.py']

model = dict(
    backbone=dict(
        pretrained=(
            'https://download.openmmlab.com/mmaction/recognition/slowonly/'
            'slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb/'
            'slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb_'
            '20210308-0d6e5a69.pth'),
        norm_cfg=dict(type='BN3d', requires_grad=True),
        non_local=((0, 0, 0), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 0, 0)),
        non_local_cfg=dict(
            sub_sample=True,
            use_scale=True,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            mode='embedded_gaussian')))
