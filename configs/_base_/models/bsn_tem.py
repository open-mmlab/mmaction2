# model settings
model = dict(
    type='TEM',
    temporal_dim=100,
    boundary_ratio=0.1,
    tem_feat_dim=400,
    tem_hidden_dim=512,
    tem_match_threshold=0.5)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
