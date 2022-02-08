_base_ = ['./slowfast_r50_4x16x1_256e_kinetics400_rgb.py']

model = dict(
    backbone=dict(
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(fusion_kernel=7)))

lr_config = dict(
    policy='step',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34,
    step=[94, 154, 196])

work_dir = './work_dirs/slowfast_r50_8x8x1_256e_kinetics400_rgb_steplr'
