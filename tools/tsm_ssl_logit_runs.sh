expname="vcops-3-us" 
config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb.py"


./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' --validate


./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' --validate


./tools/dist_train.sh $config 3 --cfg-options work_dir=/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/$expname/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' --validate

