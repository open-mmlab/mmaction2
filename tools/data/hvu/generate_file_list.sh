cd ..

# to generate file lists of videos
python hvu/generate_file_list.py --input_csv ../../data/hvu/hvu_train.csv --src_dir ../../data/hvu/videos_train \
    --output ../../data/hvu/hvu_train_video.json --mode videos
python hvu/generate_file_list.py --input_csv ../../data/hvu/hvu_val.csv --src_dir ../../data/hvu/videos_val \
    --output ../../data/hvu/hvu_val_video.json --mode videos

# to generate file list of frames
python hvu/generate_file_list.py --input_csv ../../data/hvu/hvu_train.csv --src_dir ../../data/hvu/rawframes_train \
    --output ../../data/hvu/hvu_train.json --mode frames
python hvu/generate_file_list.py --input_csv ../../data/hvu/hvu_val.csv --src_dir ../../data/hvu/rawframes_val \
    --output ../../data/hvu/hvu_val.json --mode frames
