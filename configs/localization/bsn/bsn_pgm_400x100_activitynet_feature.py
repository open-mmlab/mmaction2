# dataset settings
dataset_type = 'ImigueData'
data_root = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'
data_root_val = 'data/iMiGUE/data/tsn_feature/clip_feature_tsn_depth8_clip_length100_overlap0.5/'

ann_file_train = 'data/iMiGUE/label/imigue_clip_annotation_100_8.json'
ann_file_val = 'data/iMiGUE/label/imigue_clip_annotation_100_8.json'
ann_file_test = 'data/iMiGUE/label/imigue_clip_annotation_100_8.json'

work_dir = 'work_dirs/bsn_imigue_tsn_100_1x8_0.5/'
tem_results_dir = f'{work_dir}/tem_results/'
pgm_proposals_dir = f'{work_dir}/pgm_proposals/'
pgm_features_dir = f'{work_dir}/pgm_features/'

temporal_scale = 100
pgm_proposals_cfg = dict(
    pgm_proposals_thread=8,
    temporal_scale=temporal_scale,
    peak_threshold=0.5)
pgm_features_test_cfg = dict(
    pgm_features_thread=4,
    top_k=1000,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
pgm_features_train_cfg = dict(
    pgm_features_thread=4,
    top_k=500,
    num_sample_start=8,
    num_sample_end=8,
    num_sample_action=16,
    num_sample_interp=3,
    bsp_boundary_ratio=0.2)
