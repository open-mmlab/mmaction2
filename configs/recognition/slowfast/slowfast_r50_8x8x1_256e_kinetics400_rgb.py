_base_ = ['./slowfast_r50_4x16x1_256e_kinetics400_rgb.py']

model = dict(
    backbone=dict(
        resample_rate=4,  # tau
        speed_ratio=4,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(fusion_kernel=7)))

work_dir = './work_dirs/slowfast_r50_3d_8x8x1_256e_kinetics400_rgb'
