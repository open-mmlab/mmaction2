
# ## VCOP

# expname="tsm-vcop-3-resfrozen-us" 
# config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_vcop_resfrozen_rgb.py"

# d1_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcop/train_D1_test_D1/best_top1_acc_epoch_55.pth"

# d2_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcop/train_D2_test_D2/best_top1_acc_epoch_75.pth" 

# d3_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcop/train_D3_test_D3/best_top1_acc_epoch_60.pth"

# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from=$d1_best_ckpt find_unused_parameters=True --validate


# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from=$d2_best_ckpt find_unused_parameters=True --validate


# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from=$d3_best_ckpt find_unused_parameters=True --validate

# ## Slow fast contrastive head 

# expname="tsm-slow-fast-contrastive-head-resfrozen-us" 
# config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_slowfast_contrastivehead_ekmmsada_resfrozen_rgb.py"

# d1_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/slow_fast_contrastive_head/train_D1_test_D1/best_top1_acc_epoch_60.pth"

# d2_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/slow_fast_contrastive_head/train_D2_test_D2/best_top1_acc_epoch_45.pth" 

# d3_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/slow_fast_contrastive_head/train_D3_test_D3/best_top1_acc_epoch_85.pth"

# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from=$d1_best_ckpt find_unused_parameters=True --validate


# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from=$d2_best_ckpt find_unused_parameters=True --validate


# ./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from=$d3_best_ckpt find_unused_parameters=True --validate


## Color Jitter contrastive head 

expname="tsm-colorjitter-contrastive-head-resfrozen-us" 
config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_resfrozen_rgb.py"

d1_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/colorspatialselfsupervised/train_D1_test_D1/best_top1_acc_epoch_50.pth"

d2_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/colorspatialselfsupervised/train_D2_test_D2/best_top1_acc_epoch_100.pth" 

d3_best_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/colorspatialselfsupervised/train_D3_test_D3/best_top1_acc_epoch_85.pth"

./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from=$d1_best_ckpt find_unused_parameters=True --validate


./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from=$d2_best_ckpt find_unused_parameters=True --validate


./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from=$d3_best_ckpt find_unused_parameters=True --validate