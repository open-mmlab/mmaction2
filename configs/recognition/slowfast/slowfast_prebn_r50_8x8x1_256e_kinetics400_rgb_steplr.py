_base_ = ['./slowfast_r50_8x8x1_256e_kinetics400_rgb.py']

model = dict(backbone=dict(slow_pathway=dict(lateral_norm=True)))

lr_config = dict(
    policy='step',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=34,
    step=[94, 154, 196])

precise_bn = dict(num_iters=200, interval=5)

work_dir = './work_dirs/slowfast_prebn_r50_8x8x1_256e_kinetics400_rgb_steplr'
