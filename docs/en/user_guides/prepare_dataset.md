# Prepare Dataset

MMAction2 supports many existing datasets. In this chapter, we will lead you to prepare datasets for MMAction2.

- [Prepare Dataset](#prepare-dataset)
  - [Notes on Video Data Format](#notes-on-video-data-format)
  - [Use built-in datasets](#use-built-in-datasets)
  - [Use a custom dataset](#use-a-custom-dataset)
    - [Action Recognition](#action-recognition)
    - [Skeleton-based Action Recognition](#skeleton-based-action-recognition)
    - [Audio-based Action Recognition](#audio-based-action-recognition)
    - [Spatio-temporal Action Detection](#spatio-temporal-action-detection)
    - [Temporal Action Localization](#temporal-action-localization)
  - [Use mixed datasets for training](#use-mixed-datasets-for-training)
    - [Repeat dataset](#repeat-dataset)
  - [Browse dataset](#browse-dataset)

## Notes on Video Data Format

MMAction2 supports two types of data formats: raw frames and video. The former is widely used in previous projects such as [TSN](https://github.com/yjxiong/temporal-segment-networks).
This is fast when SSD is available but fails to scale to the fast-growing datasets.
(For example, the newest edition of [Kinetics](https://www.deepmind.com/open-source/kinetics) has 650K  videos and the total frames will take up several TBs.)
The latter saves much space but has to do the computation intensive video decoding at execution time.
To make video decoding faster, we support several efficient video loading libraries, such as [decord](https://github.com/zhreshold/decord), [PyAV](https://github.com/PyAV-Org/PyAV), etc.

## Use built-in datasets

MMAction2 already supports many datasets, we provide shell scripts for data preparation under the path `$MMACTION2/tools/data/`, please refer to [supported datasets](https://mmaction2.readthedocs.io/en/latest/datasetzoo_statistics.html) for details to prepare specific datasets.

## Use a custom dataset

The simplest way is to convert your dataset to existing dataset formats:

- `RawFrameDataset` and `VideoDataset` for [Action Recognition](#action-recognition)
- `PoseDataset` for [Skeleton-based Action Recognition](#skeleton-based-action-recognition)
- `AudioDataset` for [Audio-based Action Recognition](#Audio-based-action-recognition)
- `AVADataset` for [Spatio-temporal Action Detection](#spatio-temporal-action-detection)
- `ActivityNetDataset` for [Temporal Action Localization](#temporal-action-localization)

After the data pre-processing, the users need to further modify the config files to use the dataset.
Here is an example of using a custom dataset in rawframe format.

In `configs/task/method/my_custom_config.py`:

```python
...
# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'path/to/your/root'
data_root_val = 'path/to/your/root_val'
ann_file_train = 'data/custom/custom_train_list.txt'
ann_file_val = 'data/custom/custom_val_list.txt'
ann_file_test = 'data/custom/custom_val_list.txt'
...
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        ...),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        ...),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        ...))
...
```

### Action Recognition

There are two kinds of annotation files for action recognition.

- rawframe annotaiton for `RawFrameDataset`

  The annotation of a rawframe dataset is a text file with multiple lines,
  and each line indicates `frame_directory` (relative path) of a video,
  `total_frames` of a video and the `label` of a video, which are split by a whitespace.

  Here is an example.

  ```
  some/directory-1 163 1
  some/directory-2 122 1
  some/directory-3 258 2
  some/directory-4 234 2
  some/directory-5 295 3
  some/directory-6 121 3
  ```

- video annotation for `VideoDataset`

  The annotation of a video dataset is a text file with multiple lines,
  and each line indicates a sample video with the `filepath` (relative path) and `label`,
  which are split by a whitespace.

  Here is an example.

  ```
  some/path/000.mp4 1
  some/path/001.mp4 1
  some/path/002.mp4 2
  some/path/003.mp4 2
  some/path/004.mp4 3
  some/path/005.mp4 3
  ```

### Skeleton-based Action Recognition

The task recognizes the action class based on the skeleton sequence (time sequence of keypoints). We provide some methods to build your custom skeleton dataset.

- Build from RGB video data

  You need to extract keypoints data from video and convert it to a supported format, we provide a [tutorial](https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d/custom_dataset_training.md) with detailed instructions.

- Build from existing keypoint data

  Assuming that you already have keypoint data in coco formats, you can gather them into a pickle file.

  Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

  1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
  2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
     - `frame_dir` (str): The identifier of the corresponding video.
     - `total_frames` (int): The number of frames in this video.
     - `img_shape` (tuple\[int\]): The shape of a video frame, a tuple with two elements, in the format of `(height, width)`. Only required for 2D skeletons.
     - `original_shape` (tuple\[int\]): Same as `img_shape`.
     - `label` (int): The action label.
     - `keypoint` (np.ndarray, with shape `[M x T x V x C]`): The keypoint annotation.
       - M: number of persons;
       - T: number of frames (same as `total_frames`);
       - V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. );
       - C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
     - `keypoint_score` (np.ndarray, with shape `[M x T x V]`): The confidence score of keypoints. Only required for 2D skeletons.

  Here is an example:

  ```
  {
      "split":
          {
              'xsub_train':
                  ['S001C001P001R001A001', ...],
              'xsub_val':
                  ['S001C001P003R001A001', ...],
              ...
          }

      "annotations:
          [
              {
                  {
                      'frame_dir': 'S001C001P001R001A001',
                      'label': 0,
                      'img_shape': (1080, 1920),
                      'original_shape': (1080, 1920),
                      'total_frames': 103,
                      'keypoint': array([[[[1032. ,  334.8], ...]]])
                      'keypoint_score': array([[[0.934 , 0.9766, ...]]])
                  },
                  {
                      'frame_dir': 'S001C001P003R001A001',
                      ...
                  },
                  ...

              }
          ]
  }
  ```

  Support other keypoint formats needs further modification, please refer to [customize dataset](../advanced_guides/customize_dataset.md).

### Audio-based Action Recognition

MMAction2 provides support for audio-based action recognition tasks utilizing the `AudioDataset`. This task employs mel spectrogram features as input. An example annotation file format is as follows:

```
ihWykL5mYRI.npy 300 153
lumzQD42AN8.npy 240 321
sWFRmD9Of4s.npy 250 250
w_IpfgRsBVA.npy 300 356
```

Each line represents a training sample. Taking the first line as an example, `ihWykL5mYRI.npy` corresponds to the filename of the mel spectrogram feature. The value `300` represents the total number of frames of the original video corresponding to this mel spectrogram feature, and `153` denotes the class label. We take the following two steps to perpare the mel spectrogram feature data:

First, extract `audios` from videos:

```shell
cd $MMACTION2
python tools/data/extract_audio.py ${ROOT} ${DST_ROOT} [--ext ${EXT}] [--num-workers ${N_WORKERS}] \
    [--level ${LEVEL}]
```

- `ROOT`: The root directory of the videos.
- `DST_ROOT`: The destination root directory of the audios.
- `EXT`: Extension of the video files. e.g., `mp4`.
- `N_WORKERS`: Number of processes to be used.

Next, offline generate the `mel spectrogram features` from the audios:

```shell
cd $MMACTION2
python tools/data/build_audio_features.py ${AUDIO_HOME_PATH} ${SPECTROGRAM_SAVE_PATH} [--level ${LEVEL}] \
    [--ext $EXT] [--num-workers $N_WORKERS] [--part $PART]
```

- `AUDIO_HOME_PATH`: The root directory of the audio files.
- `SPECTROGRAM_SAVE_PATH`: The destination root directory of the audio features.
- `EXT`: Extension of the audio files. e.g., `m4a`.
- `N_WORKERS`: Number of processes to be used.
- `PART`: Determines how many parts to be splited and which part to run. e.g., `2/5` means splitting all files into 5-fold and executing the 2nd part. This is useful if you have several machines.

### Spatio-temporal Action Detection

MMAction2 supports the task based on `AVADataset`. The annotation contains groundtruth bbox and proposal bbox.

- groundtruth bbox
  groundtruth bbox is a csv file with multiple lines, and each line is a detection sample of one frame, with following formats:

  video_identifier, time_stamp, lt_x, lt_y, rb_x, rb_y, label, entity_id
  each field means:
  `video_identifier` : The identifier of the corresponding video
  `time_stamp`: The time stamp of current frame
  `lt_x`: The normalized x-coordinate of the left top point of bounding box
  `lt_y`: The normalized y-coordinate of the left top point of bounding box
  `rb_y`: The normalized x-coordinate of the right bottom point of bounding box
  `rb_y`: The normalized y-coordinate of the right bottom point of bounding box
  `label`: The action label
  `entity_id`: a unique integer allowing this box to be linked to other boxes depicting the same person in adjacent frames of this video

  Here is an example.

  ```
  _-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,12,0
  _-Z6wFjXtGQ,0902,0.063,0.049,0.524,0.996,74,0
  ...
  ```

- proposal bbox
  proposal bbox is a pickle file generated by a person detector, and usually needs to be fine-tuned on the target dataset. The pickle file contains a dict with below data structure:

  `{'video_identifier,time_stamp': bbox_info}`

  video_identifier (str): The identifier of the corresponding video
  time_stamp (int): The time stamp of current frame
  bbox_info (np.ndarray, with shape `[n, 5]`): Detected bbox, \<x1> \<y1> \<x2> \<y2> \<score>. x1, x2, y1, y2 are normalized with respect to frame size, which are between 0.0-1.0.

### Temporal Action Localization

We support Temporal Action Localization based on `ActivityNetDataset`. The annotation of ActivityNet dataset is a json file. Each key is a video name and the corresponding value is the meta data and annotation for the video.

Here is an example.

```
{
  "video1": {
      "duration_second": 211.53,
      "duration_frame": 6337,
      "annotations": [
          {
              "segment": [
                  30.025882995319815,
                  205.2318595943838
              ],
              "label": "Rock climbing"
          }
      ],
      "feature_frame": 6336,
      "fps": 30.0,
      "rfps": 29.9579255898
  },
  "video2": {...
  }
  ...
}
```

## Use mixed datasets for training

MMAction2 also supports to mix dataset for training. Currently it supports to repeat dataset.

### Repeat dataset

We use `RepeatDataset` as wrapper to repeat the dataset. For example, suppose the original dataset as `Dataset_A`,
to repeat it, the config looks like the following

```python
dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(  # This is the original config of Dataset_A
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
```

## Browse dataset

coming soon...
