_base_ = 'timesformer_spaceOnly_8x32x1_15e_8xb8_kinetics400_rgb.py'

model = dict(backbone=dict(attention_type='joint_space_time'))
