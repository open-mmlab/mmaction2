# optimizer
optimizer = dict(
    type='Adam', lr=0.01, weight_decay=0.00001)  # this lr is used for 1 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=10)
total_epochs = 20
