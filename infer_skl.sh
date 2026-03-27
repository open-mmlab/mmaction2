# python demo/demo_skeleton.py /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/lying.mp4 /home/minhtranh/works/Project/Rainscales/Lying_detection/src/demo_out.mp4 \
#     --config /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
#     --label-map /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/tools/data/skeleton/label_map_ntu60.txt \
#     --checkpoint /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/epoch_16.pth \

# # python demo/demo_enhanced_pose.py 

# VitPose-small
python demo/demo_skeleton_refactored.py /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/1732698562982.mp4 /home/minhtranh/works/Project/Rainscales/Lying_detection/src/demo_out_1736557631421.mp4 \
    --det-score-thr 0.9 \
    --config /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/best_acc_top1_epoch_24.pth \
    --label-map /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/tools/data/skeleton/label_map_ntu60.txt \
    --pose-config /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/demo_configs/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py \
    --pose-checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth

# VitPose-base-simple
# python demo/demo_skeleton_refactored.py /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/data/skeleton/Le2i/Lecture_room/video_1.avi /home/minhtranh/works/Project/Rainscales/Lying_detection/src/demo_out.mp4 \
#     --det-score-thr 0.9 \
#     --config /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
#     --checkpoint /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/work_dirs/posec3d_ntu60_2d_adam/best_acc_top1_epoch_24.pth \
#     --label-map /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/tools/data/skeleton/label_map_ntu60.txt \
#     --pose-config /home/minhtranh/works/Project/Rainscales/Lying_detection/mmaction2/demo/demo_configs/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192.py \
#     --pose-checkpoint https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base-simple_8xb64-210e_coco-256x192-0b8234ea_20230407.pth

