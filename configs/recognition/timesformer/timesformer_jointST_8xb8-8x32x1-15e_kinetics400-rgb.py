_base_ = 'timesformer_spaceOnly_8xb8-8x32x1-15e_kinetics400-rgb.py'

model = dict(backbone=dict(attention_type='joint_space_time'))
