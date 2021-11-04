# This file is used to dump all the commands to generate prediction pickle files. those
# pickle files are further used to get predictions which are correct by one model but incorrect by another 
# model 

#################################################
# # COLOR JITTER RUNS
################################################# 

d1_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D1_test_D1/best_top1_acc_epoch_50.pth"

d2_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D2_test_D2/best_top1_acc_epoch_60.pth"

d3_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D3_test_D3/best_top1_acc_epoch_45.pth"


exp_name="vcops-3-us"
config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb.py"


# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D1_test_D2/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D1]_test_[D2]/output.pkl -t 2 -w work_dirs/evaluation_pkl/$exp_name/train_D1_test_D2/videos --config-name $config --ckpt-path $d1_ckpt  -wd "Train D1 Test D2 $exp_name _ $exp_name-train-D1-test-D2" -g -gradcam


# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D1_test_D3/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D1]_test_[D3]/output.pkl -t 3 -w work_dirs/evaluation_pkl/$exp_name/train_D1_test_D3/videos --config-name $config --ckpt-path $d1_ckpt  -wd "Train D1 Test D3 $exp_name _ $exp_name-train-D1-test-D3" -g -gradcam

# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D2_test_D1/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D2]_test_[D1]/output.pkl -t 1 -w work_dirs/evaluation_pkl/$exp_name/train_D2_test_D1/videos --config-name $config --ckpt-path $d2_ckpt -wd "Train D2 Test D1 $exp_name _ $exp_name-train-D2-test-D1" -g -gradcam

# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D2_test_D3/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D2]_test_[D3]/output.pkl -t 3 -w work_dirs/evaluation_pkl/$exp_name/train_D2_test_D3/videos --config-name $config --ckpt-path $d2_ckpt  -wd "Train D2 Test D3 $exp_name _ $exp_name-train-D2-test-D3" -g -gradcam 

# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D3_test_D1/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D3]_test_[D1]/output.pkl -t 1 -w work_dirs/evaluation_pkl/$exp_name/train_D3_test_D1/videos --config-name $config --ckpt-path $d3_ckpt -wd "Train D3 Test D1 $exp_name _ $exp_name-train-D3-test-D1" -g -gradcam 

# python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D3_test_D2/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm_baseline/train_[D3]_test_[D2]/output.pkl -t 2 -w work_dirs/evaluation_pkl/$exp_name/train_D3_test_D2/videos --config-name $config --ckpt-path $d3_ckpt -wd "Train D3 Test D2 $exp_name _ $exp_name-train-D3-test-D2" -g -gradcam 

############################################################

python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D1_test_D2/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D1_test_D2/output.pkl -t 2 -w work_dirs/evaluation_pkl/$exp_name/train_D1_test_D2/videos --config-name $config --ckpt-path $d1_ckpt  -wd "Train D1 Test D2 $exp_name _ $exp_name-train-D1-test-D2" -g -gradcam


python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D1_test_D3/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D1_test_D3/output.pkl -t 3 -w work_dirs/evaluation_pkl/$exp_name/train_D1_test_D3/videos --config-name $config --ckpt-path $d1_ckpt  -wd "Train D1 Test D3 $exp_name _ $exp_name-train-D1-test-D3" -g -gradcam

python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D2_test_D1/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D2_test_D1/output.pkl -t 1 -w work_dirs/evaluation_pkl/$exp_name/train_D2_test_D1/videos --config-name $config --ckpt-path $d2_ckpt -wd "Train D2 Test D1 $exp_name _ $exp_name-train-D2-test-D1" -g -gradcam

python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D2_test_D3/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D2_test_D3/output.pkl -t 3 -w work_dirs/evaluation_pkl/$exp_name/train_D2_test_D3/videos --config-name $config --ckpt-path $d2_ckpt  -wd "Train D2 Test D3 $exp_name _ $exp_name-train-D2-test-D3" -g -gradcam 

python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D3_test_D1/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D3_test_D1/output.pkl -t 1 -w work_dirs/evaluation_pkl/$exp_name/train_D3_test_D1/videos --config-name $config --ckpt-path $d3_ckpt -wd "Train D3 Test D1 $exp_name _ $exp_name-train-D3-test-D1" -g -gradcam 

python analysis/compare_two_models.py -m1 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$exp_name/train_D3_test_D2/output.pkl -m2 /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/tsm-baseline-uniform-sampling/train_D3_test_D2/output.pkl -t 2 -w work_dirs/evaluation_pkl/$exp_name/train_D3_test_D2/videos --config-name $config --ckpt-path $d3_ckpt -wd "Train D3 Test D2 $exp_name _ $exp_name-train-D3-test-D2" -g -gradcam 