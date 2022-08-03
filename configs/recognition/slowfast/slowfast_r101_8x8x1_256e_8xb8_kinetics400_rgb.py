_base_ = ['./slowfast_r50_8x8x1_256e_8xb8_kinetics400_rgb.py']

model = dict(
    backbone=dict(slow_pathway=dict(depth=101), fast_pathway=dict(depth=101)),
    test_cfg=dict(max_testing_views=10, _delete_=True))

val_dataloader = dict(batch_size=1)
