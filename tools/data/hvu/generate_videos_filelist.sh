# to generate file lists of videos
python generate_file_list.py --input_csv ../../../data/hvu/annotations/hvu_train.csv --src_dir ../../../data/hvu/videos_train \
    --output ../../../data/hvu/hvu_train_video.json --mode videos
python generate_file_list.py --input_csv ../../../data/hvu/annotations/hvu_val.csv --src_dir ../../../data/hvu/videos_val \
    --output ../../../data/hvu/hvu_val_video.json --mode videos
