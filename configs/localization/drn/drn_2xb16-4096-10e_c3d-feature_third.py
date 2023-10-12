_base_ = ['../../_base_/default_runtime.py']

# model settings
model = dict(
    type='DRN',
    vocab_size=1301,
    feature_dim=4096,
    embed_dim=300,
    hidden_dim=512,
    bidirection=True,
    first_output_dim=256,
    fpn_feature_dim=512,
    lstm_layers=1,
    graph_node_features=1024,
    fcos_pre_nms_top_n=32,
    fcos_inference_thr=0.05,
    fcos_prior_prob=0.01,
    focal_alpha=0.25,
    focal_gamma=2.0,
    fpn_stride=[1, 2, 4],
    fcos_nms_thr=0.6,
    fcos_conv_layers=1,
    fcos_num_class=2,
    is_first_stage=False,
    is_second_stage=False)

# dataset settings
dataset_type = 'CharadesSTADataset'
root = 'data/CharadesSTA'
data_root = f'{root}/C3D_unit16_overlap0.5_merged/'
data_root_val = f'{root}/C3D_unit16_overlap0.5_merged/'
ann_file_train = f'{root}/Charades_sta_train.txt'
ann_file_val = f'{root}/Charades_sta_test.txt'
ann_file_test = f'{root}/Charades_sta_test.txt'

word2id_file = f'{root}/Charades_word2id.json'
fps_file = f'{root}/Charades_fps_dict.json'
duration_file = f'{root}/Charades_duration.json'
num_frames_file = f'{root}/Charades_frames_info.json'
window_size = 16
ft_overlap = 0.5

train_pipeline = [
    dict(
        type='PackLocalizationInputs',
        keys=('gt_bbox', 'proposals'),
        meta_keys=('vid_name', 'query_tokens', 'query_length', 'num_proposals',
                   'num_frames'))
]

val_pipeline = train_pipeline
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline,
        word2id_file=word2id_file,
        fps_file=fps_file,
        duration_file=duration_file,
        num_frames_file=num_frames_file,
        window_size=window_size,
        ft_overlap=ft_overlap),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    drop_last=True,
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root),
        pipeline=val_pipeline,
        word2id_file=word2id_file,
        fps_file=fps_file,
        duration_file=duration_file,
        num_frames_file=num_frames_file,
        window_size=window_size,
        ft_overlap=ft_overlap),
)
test_dataloader = val_dataloader

max_epochs = 10
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=1,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

val_evaluator = dict(type='RecallatTopK', topK_list=(1, 5), threshold=0.5)
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=1e-6),
    clip_grad=dict(max_norm=5, norm_type=2))

find_unused_parameters = True
