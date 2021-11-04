d1_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D1_test_D1/best_top1_acc_epoch_50.pth"

d2_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D2_test_D2/best_top1_acc_epoch_60.pth"

d3_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/vcops-3-us/train_D3_test_D3/best_top1_acc_epoch_45.pth"


expname="vcops-3-us"
config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_vcop_rgb.py"

python tools/test.py $config $d1_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D1_test_D2/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D1_test_D3/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D2_test_D1/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D2_test_D3/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D3_test_D1/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D1'


python tools/test.py /home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb.py $d3_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D3_test_D2/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D2'