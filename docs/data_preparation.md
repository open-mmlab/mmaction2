# Data Preparation

## Notes on Video Data Format

MMAction2 supports two types of data format: raw frames and video. The former is widely used in previous projects such as [TSN](https://github.com/yjxiong/temporal-segment-networks).
This is fast when SSD is available but fails to scale to the fast-growing datasets.
(For example, the newest edition of [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) has 650K  videos and the total frames will take up several TBs.)
The latter saves much space but has to do the computation intensive video decoding at execution time
To make video decoding faster, we support several efficient video loading libraries, such as [decord](https://github.com/zhreshold/decord), [PyAV](https://github.com/PyAV-Org/PyAV), etc.

## Getting Data

The following guide is helpful when you want to experiment with custom dataset.
Similar to the datasets stated above, it is recommended organizing in `$MMACTION2/data/$DATASET`.

### Prepare videos

 Please refer to the official website and/or the official script to prepare the videos.
Note that the videos should be arranged in either

(1). A two-level directory organized by `${CLASS_NAME}/${VIDEO_ID}`, which is recommended to be used for for action recognition datasets (such as UCF101 and Kinetics)

(2). A single-level directory, which is recommended to be used for for action detection datasets or those with multiple annotations per video (such as THUMOS14).

### Extract frames

To extract both frames and optical flow, you can use the tool [denseflow](https://github.com/open-mmlab/denseflow) we wrote.
Since different frame extraction tools produce different number of frames,
it is beneficial to use the same tool to do both frame extraction and the flow computation, to avoid mismatching of frame counts.

```shell
python build_rawframes.py ${SRC_FOLDER} ${OUT_FOLDER} [--task ${TASK}] [--level ${LEVEL}] \
    [--num-worker ${NUM_WORKER}] [--flow-type ${FLOW_TYPE}] [--out-format ${OUT_FORMAT}] \
    [--ext ${EXT}] [--new-width ${NEW_WIDTH}] [--new-height ${NEW_HEIGHT}] [--new-short ${NEW_SHORT}]
    [--resume] [--use-opencv]
```

- `SRC_FOLDER`: Folder of the original video.
- `OUT_FOLDER`: Root folder where the extracted frames and optical flow store.
- `TASK`: Extraction task indicating which kind of frames to extract. Allowed choices are `rgb`, `flow`, `both`.
- `LEVEL`: Directory level. 1 for the single-level directory or 2 for the two-level directory.
- `NUM_WORKER`: Number of workers to build rawframes.
- `FLOW_TYPE`: Flow type to extract, e.g., `None`, `tvl1`, `warp_tvl1`, `farn`, `brox`.
- `OUT_FORMAT`: Output format for extracted frames, e.g., `jpg`, `h5`, `png`.
- `EXT`: Video file extension, e.g., `avi`, `mp4`.
- `NEW_WIDTH`: Resized image width of output.
- `NEW_HEIGHT`: Resized image height of output.
- `NEW_SHORT`: Resized image short side length keeping ratio.
- `--resume`: Whether to resume optical flow extraction instead of overwriting.
- `--use-opencv`: Whether to use OpenCV to extract rgb frames.

The recommended practice is

1. set `$OUT_FOLDER` to be a folder located in SSD.
2. symlink the link `$OUT_FOLDER` to `$MMACTION2/data/$DATASET/rawframes`.

```shell
ln -s ${YOUR_FOLDER} $MMACTION2/data/$DATASET/rawframes
```

#### Alternative to denseflow

In case your device doesn't fulfill the installation requirement of [denseflow](https://github.com/open-mmlab/denseflow)(like Nvidia driver version), or you just want to see some quick demos about flow extraction, we provide a python script `tools/flow_extraction.py` as an alternative to denseflow. You can use it for rgb frames and optical flow extraction from one or several videos. Note that the speed of the script is much slower than denseflow, since it runs optical flow algorithms on CPU.

```shell
python tools/flow_extraction.py --input ${INPUT} [--prefix ${PREFIX}] [--dest ${DEST}] [--rgb-tmpl ${RGB_TMPL}] \
    [--flow-tmpl ${FLOW_TMPL}] [--start-idx ${START_IDX}] [--method ${METHOD}] [--bound ${BOUND}] [--save-rgb]
```

- `INPUT`:  Videos for frame extraction, can be single video or a video list, the video list should be a txt file and just consists of filenames without directories.
- `PREFIX`: The prefix of input videos, used when input is a video list.
- `DEST`: The destination to save extracted frames.
- `RGB_TMPL`: The template filename of rgb frames.
- `FLOW_TMPL`: The template filename of flow frames.
- `START_IDX`: The start index of extracted frames.
- `METHOD`: The method used to generate flow.
- `BOUND`: The maximum of optical flow.
- `SAVE_RGB`: Also save extracted rgb frames.

### Generate file list

We provide a convenient script to generate annotation file list. You can use the following command to extract frames.

```shell
cd $MMACTION2
python tools/data/build_file_list.py ${DATASET} ${SRC_FOLDER} [--rgb-prefix ${RGB_PREFIX}] \
    [--flow-x-prefix ${FLOW_X_PREFIX}] [--flow-y-prefix ${FLOW_Y_PREFIX}] [--num-split ${NUM_SPLIT}] \
    [--subset ${SUBSET}] [--level ${LEVEL}] [--format ${FORMAT}] [--out-root-path ${OUT_ROOT_PATH}] \
    [--shuffle]
```

- `DATASET`: Dataset to be prepared, e.g., `ucf101`, `kinetics400`, `thumos14`, `sthv1`, `sthv2`, etc.
- `SRC_FOLDER`: Folder of the corresponding data format:
  - "$MMACTION2/data/$DATASET/rawframes" if `--format rawframes`.
  - "$MMACTION2/data/$DATASET/videos" if `--format videos`.
- `RGB_PREFIX`: Name prefix of rgb frames.
- `FLOW_X_PREFIX`: Name prefix of x flow frames.
- `FLOW_Y_PREFIX`: Name prefix of y flow frames.
- `NUM_SPLIT`: Number of split to file list.
- `SUBSET`: Subset to generate file list. Allowed choice are `train`, `val`, `test`.
- `LEVEL`: Directory level. 1 for the single-level directory or 2 for the two-level directory.
- `FORMAT`: Source data format to generate file list. Allowed choices are `rawframes`, `videos`.
- `OUT_ROOT_PATH`: Root path for output
- `--shuffle`: Whether to shuffle the file list.

Now, you can go to [getting_started.md](getting_started.md) to train and test the model.
