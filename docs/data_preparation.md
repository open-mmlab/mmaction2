# Data Preparation

## Notes on Video Data Format

MMAction2 supports two types of data format: raw frames and video. The former is widely used in previous projects such as [TSN](https://github.com/yjxiong/temporal-segment-networks).
This is fast when SSD is available but fails to scale to the fast-growing datasets.
(For example, the newest edition of [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) has 650K  videos and the total frames will take up several TBs.)
The latter saves much space but has to do the computation intensive video decoding at execution time
To make video decoding faster, we support several efficient video loading libraries, such as [decord](https://github.com/zhreshold/decord), [PyAV](https://github.com/PyAV-Org/PyAV), etc.

## Supported Datasets

The supported datasets are listed below.
We provide shell scripts for data preparation under the path `$MMACTION2/tools/data/`.
To ease usage, we provide tutorials of data deployment for each dataset.

- [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/): See [preparing_ucf101.md](/tools/data/ucf101/preparing_ucf101.md).
- [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/): See [preparing_hmdb51.md](/tools/data/hmdb51/preparing_hmdb51.md).
- [Kinetics400](https://deepmind.com/research/open-source/kinetics): See [preparing_kinetics400.md](/tools/data/kinetics400/preparing_kinetics400.md)
- [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/download.html): See [preparing_thumos14.md](/tools/data/thumos14/preparing_thumos14.md)
- [Something-Something V1](https://20bn.com/datasets/something-something/v1): See [preparing_sthv1.md](/tools/data/sthv1/preparing_sthv1.md)
- [Something-Something V2](https://20bn.com/datasets/something-something): See [preparing_sthv2.md](/tools/data/sthv2/preparing_sthv2.md)
- [Moments in Time](http://moments.csail.mit.edu/): See [preparing_mit.md](/tools/data/mit/preparing_mit.md)
- [Multi-Moments in Time](http://moments.csail.mit.edu/challenge_iccv_2019.html): See [preparing_mmit.md](/tools/data/mmit/preparing_mmit.md)
- ActivityNet_feature: See [praparing_activitynet.md](/tools/data/activitynet/preparing_activitynet.md)

Now, you can switch to [getting_started.md](getting_started.md) to train and test the model.

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
