d1_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm-colorjitter-contrastive-head-resfrozen-us/train_D1_test_D1/best_top1_acc_epoch_25.pth"

d2_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm-colorjitter-contrastive-head-resfrozen-us/train_D2_test_D2/best_top1_acc_epoch_5.pth"

d3_ckpt="/home/ubuntu/users/maiti/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/tsm-colorjitter-contrastive-head-resfrozen-us/train_D3_test_D3/best_top1_acc_epoch_15.pth"


expname="tsm-colorjitter-contrastive-head-resfrozen-us"
config="/home/ubuntu/users/maiti/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_resfrozen_rgb.py"

python tools/test.py $config $d1_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D1_test_D2/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D1_test_D3/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D2_test_D1/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D2_test_D3/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D3_test_D1/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D1'


python tools/test.py $config $d3_ckpt --out /home/ubuntu/users/maiti/projects/mmaction2/work_dirs/evaluation_pkl/$expname/train_D3_test_D2/output.pkl --eval top_k_accuracy --cfg-options data.test.domain='D2'