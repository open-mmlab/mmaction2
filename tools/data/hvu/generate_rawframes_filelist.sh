# to generate file list of frames
python generate_file_list.py --input_csv ../../../data/hvu/annotations/hvu_train.csv --src_dir ../../../data/hvu/rawframes_train \
    --output ../../../data/hvu/hvu_train.json --mode frames
python generate_file_list.py --input_csv ../../../data/hvu/annotations/hvu_val.csv --src_dir ../../../data/hvu/rawframes_val \
    --output ../../../data/hvu/hvu_val.json --mode frames
