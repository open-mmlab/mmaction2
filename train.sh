export CUBLAS_WORKSPACE_CONFIG=:16:8

python  /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/tools/train.py /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --work-dir work_dirs/posec3d_ntu60_2d_adam/ \
    --seed 0 \
    --amp 