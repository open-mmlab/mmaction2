# dataset settings
dataset_type = 'ActivityNetDataset'
data_root = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
data_root_val = 'data/ActivityNet/activitynet_feature_cuhk/csv_mean_100/'
ann_file_train = 'data/ActivityNet/anet_anno_train.json'
ann_file_val = 'data/ActivityNet/anet_anno_val.json'
ann_file_test = 'data/ActivityNet/anet_anno_test.json'

work_dir = 'work_dirs/bsn_400x100_20e_1x16_activitynet_feature/'
tem_results_dir = f'{work_dir}/tem_results/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

temporal_scale = 100
pgm_proposals_cfg = dict(
    pgm_proposals_thread=8, temporal_scale=temporal_scale, peak_threshold=0.5)
pgm_features_test_cfg = dict(
    pgm_features_thread=32,
    top_k=1000,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
pgm_features_train_cfg = dict(
    pgm_features_thread=32,
    top_k=500,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
