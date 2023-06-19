# Copyright (c) OpenMMLab. All rights reserved.
_base_ = [
    'mmaction::detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb.py'  # noqa: E501
]

proposal_file_train = 'data/multisports/annotations/multisports_proposals_train.pkl'  # noqa: E501
proposal_file_val = 'data/multisports/annotations/multisports_proposals_val.pkl'  # noqa: E501

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(proposal_file=proposal_file_train))

val_dataloader = dict(
    num_workers=2, dataset=dict(proposal_file=proposal_file_val))

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01))

load_from = 'https://download.openmmlab.com/mmaction/v1.0/detection/slowonly/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb/slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-8e_multisports-rgb_20230320-a1ca5e76.pth'  # noqa: E501
