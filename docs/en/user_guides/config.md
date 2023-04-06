# Learn about Configs

We use python files as configs, incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.
You can find all the provided configs under `$MMAction2/configs`. If you wish to inspect the config file,
you may run `python tools/analysis_tools/print_config.py /PATH/TO/CONFIG` to see the complete config.

<!-- TOC -->

- [Learn about Configs](#learn-about-configs)
  - [Modify config through script arguments](#modify-config-through-script-arguments)
  - [Config File Structure](#config-file-structure)
  - [Config File Naming Convention](#config-file-naming-convention)
    - [Config System for Action Recognition](#config-system-for-action-recognition)
    - [Config System for Spatio-Temporal Action Detection](#config-system-for-spatio-temporal-action-detection)
    - [Config System for Action localization](#config-system-for-action-localization)

<!-- TOC -->

## Modify config through script arguments

When submitting jobs using `tools/train.py` or `tools/test.py`, you may specify `--cfg-options` to in-place modify the config.

- Update config keys of dict.

  The config options can be specified following the order of the dict keys in the original config.
  For example, `--cfg-options model.backbone.norm_eval=False` changes the all BN modules in model backbones to `train` mode.

- Update keys inside a list of configs.

  Some config dicts are composed as a list in your config. For example, the training pipeline `train_pipeline` is normally a list
  e.g. `[dict(type='SampleFrames'), ...]`. If you want to change `'SampleFrames'` to `'DenseSampleFrames'` in the pipeline,
  you may specify `--cfg-options train_pipeline.0.type=DenseSampleFrames`.

- Update values of list/tuples.

  If the value to be updated is a list or a tuple. For example, the config file normally sets `model.data_preprocessor.mean=[123.675, 116.28, 103.53]`. If you want to
  change this key, you may specify `--cfg-options model.data_preprocessor.mean="[128,128,128]"`. Note that the quotation mark " is necessary to support list/tuple data types.

## Config File Structure

There are 3 basic component types under `configs/_base_`, models, schedules, default_runtime.
Many methods could be easily constructed with one of each like TSN, I3D, SlowOnly, etc.
The configs that are composed by components from `_base_` are called _primitive_.

For all configs under the same folder, it is recommended to have only **one** _primitive_ config. All other configs should inherit from the _primitive_ config. In this way, the maximum of inheritance level is 3.

For easy understanding, we recommend contributors to inherit from exiting methods.
For example, if some modification is made based on TSN, users may first inherit the basic TSN structure by specifying `_base_ = ../tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py`, then modify the necessary fields in the config files.

If you are building an entirely new method that does not share the structure with any of the existing methods, you may create a folder under `configs/TASK`.

Please refer to [mmengine](https://mmengine.readthedocs.io/en/latest/tutorials/config.html) for detailed documentation.

## Config File Naming Convention

We follow the style below to name config files. Contributors are advised to follow the same style. The config file names are divided into several parts. Logically, different parts are concatenated by underscores `'_'`, and settings in the same part are concatenated by dashes `'-'`.

```
{algorithm info}_{module info}_{training info}_{data info}.py
```

`{xxx}` is required field and `[yyy]` is optional.

- `{algorithm info}`:
  - `{model}`: model type, e.g. `tsn`, `i3d`, `swin`, `vit`, etc.
  - `[model setting]`: specific setting for some models, e.g. `base`, `p16`, `w877`, etc.
- `{module info}`:
  - `[pretained info]`: pretrained information, e.g. `kinetics400-pretrained`, `in1k-pre`, etc.
  - `{backbone}`: backbone type. e.g. `r50` (ResNet-50), etc.
  - `[backbone setting]`: specific setting for some backbones, e.g. `nl-dot-product`, `bnfrozen`, `nopool`, etc.
- `{training info}`:
  - `{gpu x batch_per_gpu]}`: GPUs and samples per GPU.
  - `{pipeline setting}`: frame sample setting, e.g. `dense`, `{clip_len}x{frame_interval}x{num_clips}`, `u48`, etc.
  - `{schedule}`: training schedule, e.g. `coslr-20e`.
- `{data info}`:
  - `{dataset}`: dataset name, e.g. `kinetics400`, `mmit`, etc.
  - `{modality}`: data modality, e.g. `rgb`, `flow`, `keypoint-2d`, etc.

### Config System for Action Recognition

We incorporate modular design into our config system,
which is convenient to conduct various experiments.

- An Example of TSN

  To help the users have a basic idea of a complete config structure and the modules in an action recognition system,
  we make brief comments on the config of TSN as the following.
  For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.

  ```python
  # model settings
  model = dict(  # Config of the model
      type='Recognizer2D',  # Class name of the recognizer
      backbone=dict(  # Dict for backbone
          type='ResNet',  # Name of the backbone
          pretrained='torchvision://resnet50',  # The url/site of the pretrained model
          depth=50,  # Depth of ResNet model
          norm_eval=False),  # Whether to set BN layers to eval mode when training
      cls_head=dict(  # Dict for classification head
          type='TSNHead',  # Name of classification head
          num_classes=400,  # Number of classes to be classified.
          in_channels=2048,  # The input channels of classification head.
          spatial_type='avg',  # Type of pooling in spatial dimension
          consensus=dict(type='AvgConsensus', dim=1),  # Config of consensus module
          dropout_ratio=0.4,  # Probability in dropout layer
          init_std=0.01, # Std value for linear layer initiation
          average_clips='prob'),  # Method to average multiple clip results
      data_preprocessor=dict(  # Dict for data preprocessor
          type='ActionDataPreprocessor',  # Name of data preprocessor
          mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
          std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
          format_shape='NCHW'),  # Final image shape format
      # model training and testing settings
      train_cfg=None,  # Config of training hyperparameters for TSN
      test_cfg=None)  # Config for testing hyperparameters for TSN.

  # dataset settings
  dataset_type = 'RawframeDataset'  # Type of dataset for training, validation and testing
  data_root = 'data/kinetics400/rawframes_train/'  # Root path to data for training
  data_root_val = 'data/kinetics400/rawframes_val/'  # Root path to data for validation and testing
  ann_file_train = 'data/kinetics400/kinetics400_train_list_rawframes.txt'  # Path to the annotation file for training
  ann_file_val = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for validation
  ann_file_test = 'data/kinetics400/kinetics400_val_list_rawframes.txt'  # Path to the annotation file for testing

  train_pipeline = [  # Training data processing pipeline
      dict(  # Config of SampleFrames
          type='SampleFrames',  # Sample frames pipeline, sampling frames from video
          clip_len=1,  # Frames of each sampled output clip
          frame_interval=1,  # Temporal interval of adjacent sampled frames
          num_clips=3),  # Number of clips to be sampled
      dict(  # Config of RawFrameDecode
          type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
      dict(  # Config of Resize
          type='Resize',  # Resize pipeline
          scale=(-1, 256)),  # The scale to resize images
      dict(  # Config of MultiScaleCrop
          type='MultiScaleCrop',  # Multi scale crop pipeline, cropping images with a list of randomly selected scales
          input_size=224,  # Input size of the network
          scales=(1, 0.875, 0.75, 0.66),  # Scales of width and height to be selected
          random_crop=False,  # Whether to randomly sample cropping bbox
          max_wh_scale_gap=1),  # Maximum gap of w and h scale levels
      dict(  # Config of Resize
          type='Resize',  # Resize pipeline
          scale=(224, 224),  # The scale to resize images
          keep_ratio=False),  # Whether to resize with changing the aspect ratio
      dict(  # Config of Flip
          type='Flip',  # Flip Pipeline
          flip_ratio=0.5),  # Probability of implementing flip
      dict(  # Config of FormatShape
          type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
          input_format='NCHW'),  # Final image shape format
      dict(type='PackActionInputs')  # Config of PackActionInputs
  ]
  val_pipeline = [  # Validation data processing pipeline
      dict(  # Config of SampleFrames
          type='SampleFrames',  # Sample frames pipeline, sampling frames from video
          clip_len=1,  # Frames of each sampled output clip
          frame_interval=1,  # Temporal interval of adjacent sampled frames
          num_clips=3,  # Number of clips to be sampled
          test_mode=True),  # Whether to set test mode in sampling
      dict(  # Config of RawFrameDecode
          type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
      dict(  # Config of Resize
          type='Resize',  # Resize pipeline
          scale=(-1, 256)),  # The scale to resize images
      dict(  # Config of CenterCrop
          type='CenterCrop',  # Center crop pipeline, cropping the center area from images
          crop_size=224),  # The size to crop images
      dict(  # Config of Flip
          type='Flip',  # Flip pipeline
          flip_ratio=0),  # Probability of implementing flip
      dict(  # Config of FormatShape
          type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
          input_format='NCHW'),  # Final image shape format
      dict(type='PackActionInputs')  # Config of PackActionInputs
  ]
  test_pipeline = [  # Testing data processing pipeline
      dict(  # Config of SampleFrames
          type='SampleFrames',  # Sample frames pipeline, sampling frames from video
          clip_len=1,  # Frames of each sampled output clip
          frame_interval=1,  # Temporal interval of adjacent sampled frames
          num_clips=25,  # Number of clips to be sampled
          test_mode=True),  # Whether to set test mode in sampling
      dict(  # Config of RawFrameDecode
          type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
      dict(  # Config of Resize
          type='Resize',  # Resize pipeline
          scale=(-1, 256)),  # The scale to resize images
      dict(  # Config of TenCrop
          type='TenCrop',  # Ten crop pipeline, cropping ten area from images
          crop_size=224),  # The size to crop images
      dict(  # Config of Flip
          type='Flip',  # Flip pipeline
          flip_ratio=0),  # Probability of implementing flip
      dict(  # Config of FormatShape
          type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
          input_format='NCHW'),  # Final image shape format
      dict(type='PackActionInputs')  # Config of PackActionInputs
  ]

  train_dataloader = dict(  # Config of train dataloader
      batch_size=32,  # Batch size of each single GPU during training
      num_workers=8,  # Workers to pre-fetch data for each single GPU during training
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
      sampler=dict(
          type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # Randomly shuffle the training data in each epoch
      dataset=dict(  # Config of train dataset
          type=dataset_type,
          ann_file=ann_file_train,  # Path of annotation file
          data_prefix=dict(img=data_root),  # Prefix of frame path
          pipeline=train_pipeline))
  val_dataloader = dict(  # Config of validation dataloader
      batch_size=1,  # Batch size of each single GPU during validation
      num_workers=8,  # Workers to pre-fetch data for each single GPU during validation
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # Not shuffle during validation and testing
      dataset=dict(  # Config of validation dataset
          type=dataset_type,
          ann_file=ann_file_val,  # Path of annotation file
          data_prefix=dict(img=data_root_val),  # Prefix of frame path
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = dict(  # Config of test dataloader
      batch_size=32,  # Batch size of each single GPU during testing
      num_workers=8,  # Workers to pre-fetch data for each single GPU during testing
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # Not shuffle during validation and testing
      dataset=dict(  # Config of test dataset
          type=dataset_type,
          ann_file=ann_file_val,  # Path of annotation file
          data_prefix=dict(img=data_root_val),  # Prefix of frame path
          pipeline=test_pipeline,
          test_mode=True))

  # evaluation settings
  val_evaluator = dict(type='AccMetric')  # Config of validation evaluator
  test_evaluator = val_evaluator  # Config of testing evaluator

  train_cfg = dict(  # Config of training loop
      type='EpochBasedTrainLoop',  # Name of training loop
      max_epochs=100,  # Total training epochs
      val_begin=1,  # The epoch that begins validating
      val_interval=1)  # Validation interval
  val_cfg = dict(  # Config of validation loop
      type='ValLoop')  # Name of validation loop
  test_cfg = dict( # Config of testing loop
      type='TestLoop')  # Name of testing loop

  # learning policy
  param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
      dict(type='MultiStepLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
          begin=0,  # Step at which to start updating the learning rate
          end=100,  # Step at which to stop updating the learning rate
          by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
          milestones=[40, 80],  # Steps to decay the learning rate
          gamma=0.1)]  # Multiplicative factor of learning rate decay

  # optimizer
  optim_wrapper = dict(  # Config of optimizer wrapper
      type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
      optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
          type='SGD',  # Name of optimizer
          lr=0.01,  # Learning rate
          momentum=0.9,  # Momentum factor
          weight_decay=0.0001),  # Weight decay
      clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip

  # runtime settings
  default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
      runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
      timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
      logger=dict(
          type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
          interval=20,  # Interval to print the log
          ignore_last=False), # Ignore the log of last iterations in each epoch
      param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
      checkpoint=dict(
          type='CheckpointHook',  # The hook to save checkpoints periodically
          interval=3,  # The saving period
          save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
          max_keep_ckpts=3),  # The maximum checkpoints to keep
      sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
      sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
  env_cfg = dict(  # Dict for setting environment
      cudnn_benchmark=False,  # Whether to enable cudnn benchmark
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # Parameters to setup multiprocessing
      dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

  log_processor = dict(
      type='LogProcessor',  # Log processor used to format log information
      window_size=20,  # Default smooth interval
      by_epoch=True)  # Whether to format logs with epoch type
  vis_backends = [  # List of visualization backends
      dict(type='LocalVisBackend')]  # Local visualization backend
  visualizer = dict(  # Config of visualizer
      type='ActionVisualizer',  # Name of visualizer
      vis_backends=vis_backends)
  log_level = 'INFO'  # The level of logging
  load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
  resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
  ```

### Config System for Spatio-Temporal Action Detection

We incorporate modular design into our config system, which is convenient to conduct various experiments.

- An Example of FastRCNN

  To help the users have a basic idea of a complete config structure and the modules in a spatio-temporal action detection system,
  we make brief comments on the config of FastRCNN as the following.
  For more detailed usage and alternative for per parameter in each module, please refer to the API documentation.

  ```python
  # model setting
  model = dict(  # Config of the model
      type='FastRCNN',  # Class name of the detector
      _scope_='mmdet',  # The scope of current config
      backbone=dict(  # Dict for backbone
          type='ResNet3dSlowOnly',  # Name of the backbone
          depth=50, # Depth of ResNet model
          pretrained=None,   # The url/site of the pretrained model
          pretrained2d=False, # If the pretrained model is 2D
          lateral=False,  # If the backbone is with lateral connections
          num_stages=4, # Stages of ResNet model
          conv1_kernel=(1, 7, 7), # Conv1 kernel size
          conv1_stride_t=1, # Conv1 temporal stride
          pool1_stride_t=1, # Pool1 temporal stride
          spatial_strides=(1, 2, 2, 1)),  # The spatial stride for each ResNet stage
      roi_head=dict(  # Dict for roi_head
          type='AVARoIHead',  # Name of the roi_head
          bbox_roi_extractor=dict(  # Dict for bbox_roi_extractor
              type='SingleRoIExtractor3D',  # Name of the bbox_roi_extractor
              roi_layer_type='RoIAlign',  # Type of the RoI op
              output_size=8,  # Output feature size of the RoI op
              with_temporal_pool=True), # If temporal dim is pooled
          bbox_head=dict( # Dict for bbox_head
              type='BBoxHeadAVA', # Name of the bbox_head
              in_channels=2048, # Number of channels of the input feature
              num_classes=81, # Number of action classes + 1
              multilabel=True,  # If the dataset is multilabel
              dropout_ratio=0.5),  # The dropout ratio used
      data_preprocessor=dict(  # Dict for data preprocessor
          type='ActionDataPreprocessor',  # Name of data preprocessor
          mean=[123.675, 116.28, 103.53],  # Mean values of different channels to normalize
          std=[58.395, 57.12, 57.375],  # Std values of different channels to normalize
          format_shape='NCHW')),  # Final image shape format
      # model training and testing settings
      train_cfg=dict(  # Training config of FastRCNN
          rcnn=dict(  # Dict for rcnn training config
              assigner=dict(  # Dict for assigner
                  type='MaxIoUAssignerAVA', # Name of the assigner
                  pos_iou_thr=0.9,  # IoU threshold for positive examples, > pos_iou_thr -> positive
                  neg_iou_thr=0.9,  # IoU threshold for negative examples, < neg_iou_thr -> negative
                  min_pos_iou=0.9), # Minimum acceptable IoU for positive examples
              sampler=dict( # Dict for sample
                  type='RandomSampler', # Name of the sampler
                  num=32, # Batch Size of the sampler
                  pos_fraction=1, # Positive bbox fraction of the sampler
                  neg_pos_ub=-1,  # Upper bound of the ratio of num negative to num positive
                  add_gt_as_proposals=True), # Add gt bboxes as proposals
              pos_weight=1.0)),  # Loss weight of positive examples
      test_cfg=dict(rcnn=None))  # Testing config of FastRCNN

  # dataset settings
  dataset_type = 'AVADataset' # Type of dataset for training, validation and testing
  data_root = 'data/ava/rawframes'  # Root path to data
  anno_root = 'data/ava/annotations'  # Root path to annotations

  ann_file_train = f'{anno_root}/ava_train_v2.1.csv'  # Path to the annotation file for training
  ann_file_val = f'{anno_root}/ava_val_v2.1.csv'  # Path to the annotation file for validation

  exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.1.csv'  # Path to the exclude annotation file for training
  exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.1.csv'  # Path to the exclude annotation file for validation

  label_file = f'{anno_root}/ava_action_list_v2.1_for_activitynet_2018.pbtxt'  # Path to the label file

  proposal_file_train = f'{anno_root}/ava_dense_proposals_train.FAIR.recall_93.9.pkl'  # Path to the human detection proposals for training examples
  proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'  # Path to the human detection proposals for validation examples

  train_pipeline = [  # Training data processing pipeline
      dict(  # Config of SampleFrames
          type='AVASampleFrames',  # Sample frames pipeline, sampling frames from video
          clip_len=4,  # Frames of each sampled output clip
          frame_interval=16),  # Temporal interval of adjacent sampled frames
      dict(  # Config of RawFrameDecode
          type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
      dict(  # Config of RandomRescale
          type='RandomRescale',   # Randomly rescale the shortedge by a given range
          scale_range=(256, 320)),   # The shortedge size range of RandomRescale
      dict(  # Config of RandomCrop
          type='RandomCrop',   # Randomly crop a patch with the given size
          size=256),   # The size of the cropped patch
      dict(  # Config of Flip
          type='Flip',  # Flip Pipeline
          flip_ratio=0.5),  # Probability of implementing flip
      dict(  # Config of FormatShape
          type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
          input_format='NCTHW',  # Final image shape format
          collapse=True),   # Collapse the dim N if N == 1
      dict(type='PackActionInputs') # Pack input data
  ]

  val_pipeline = [  # Validation data processing pipeline
      dict(  # Config of SampleFrames
          type='AVASampleFrames',  # Sample frames pipeline, sampling frames from video
          clip_len=4,  # Frames of each sampled output clip
          frame_interval=16),  # Temporal interval of adjacent sampled frames
      dict(  # Config of RawFrameDecode
          type='RawFrameDecode'),  # Load and decode Frames pipeline, picking raw frames with given indices
      dict(  # Config of Resize
          type='Resize',  # Resize pipeline
          scale=(-1, 256)),  # The scale to resize images
      dict(  # Config of FormatShape
          type='FormatShape',  # Format shape pipeline, Format final image shape to the given input_format
          input_format='NCTHW',  # Final image shape format
          collapse=True),   # Collapse the dim N if N == 1
      dict(type='PackActionInputs') # Pack input data
  ]

  train_dataloader = dict(  # Config of train dataloader
      batch_size=32,  # Batch size of each single GPU during training
      num_workers=8,  # Workers to pre-fetch data for each single GPU during training
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
      sampler=dict(
          type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # Randomly shuffle the training data in each epoch
      dataset=dict(  # Config of train dataset
          type=dataset_type,
          ann_file=ann_file_train,  # Path of annotation file
          exclude_file=exclude_file_train,  # Path of exclude annotation file
          label_file=label_file,  # Path of label file
          data_prefix=dict(img=data_root),  # Prefix of frame path
          proposal_file=proposal_file_train,  # Path of human detection proposals
          pipeline=train_pipeline))
  val_dataloader = dict(  # Config of validation dataloader
      batch_size=1,  # Batch size of each single GPU during evaluation
      num_workers=8,  # Workers to pre-fetch data for each single GPU during evaluation
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # Not shuffle during validation and testing
      dataset=dict(  # Config of validation dataset
          type=dataset_type,
          ann_file=ann_file_val,  # Path of annotation file
          exclude_file=exclude_file_val,  # Path of exclude annotation file
          label_file=label_file,  # Path of label file
          data_prefix=dict(img=data_root_val),  # Prefix of frame path
          proposal_file=proposal_file_val,  # Path of human detection proposals
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = val_dataloader  # Config of testing dataloader

  # evaluation settings
  val_evaluator = dict(  # Config of validation evaluator
      type='AVAMetric',
      ann_file=ann_file_val,
      label_file=label_file,
      exclude_file=exclude_file_val)
  test_evaluator = val_evaluator  # Config of testing evaluator

  train_cfg = dict(  # Config of training loop
      type='EpochBasedTrainLoop',  # Name of training loop
      max_epochs=20,  # Total training epochs
      val_begin=1,  # The epoch that begins validating
      val_interval=1)  # Validation interval
  val_cfg = dict(  # Config of validation loop
      type='ValLoop')  # Name of validation loop
  test_cfg = dict( # Config of testing loop
      type='TestLoop')  # Name of testing loop

  # learning policy
  param_scheduler = [ # Parameter scheduler for updating optimizer parameters, support dict or list
      dict(type='LinearLR',  # Decays the learning rate of each parameter group by linearly changing small multiplicative factor
          start_factor=0.1,  # The number we multiply learning rate in the first epoch
          by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
    	  begin=0,  # Step at which to start updating the learning rate
    	  end=5),  # Step at which to stop updating the learning rate
      dict(type='MultiStepLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
          begin=0,  # Step at which to start updating the learning rate
          end=20,  # Step at which to stop updating the learning rate
          by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
          milestones=[10, 15],  # Steps to decay the learning rate
          gamma=0.1)]  # Multiplicative factor of learning rate decay

  # optimizer
  optim_wrapper = dict(  # Config of optimizer wrapper
      type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
      optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
          type='SGD',  # Name of optimizer
          lr=0.2,  # Learning rate
          momentum=0.9,  # Momentum factor
          weight_decay=0.0001),  # Weight decay
      clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip

  # runtime settings
  default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
      runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
      timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
      logger=dict(
          type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
          interval=20,  # Interval to print the log
          ignore_last=False), # Ignore the log of last iterations in each epoch
      param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
      checkpoint=dict(
          type='CheckpointHook',  # The hook to save checkpoints periodically
          interval=3,  # The saving period
          save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
          max_keep_ckpts=3),  # The maximum checkpoints to keep
      sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
      sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
  env_cfg = dict(  # Dict for setting environment
      cudnn_benchmark=False,  # Whether to enable cudnn benchmark
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # Parameters to setup multiprocessing
      dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

  log_processor = dict(
      type='LogProcessor',  # Log processor used to format log information
      window_size=20,  # Default smooth interval
      by_epoch=True)  # Whether to format logs with epoch type
  vis_backends = [  # List of visualization backends
      dict(type='LocalVisBackend')]  # Local visualization backend
  visualizer = dict(  # Config of visualizer
      type='ActionVisualizer',  # Name of visualizer
      vis_backends=vis_backends)
  log_level = 'INFO'  # The level of logging
  load_from = ('https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/'
               'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb/'
               'slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb_20220901-e7b65fad.pth')  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
  resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
  ```

### Config System for Action localization

We incorporate modular design into our config system,
which is convenient to conduct various experiments.

- An Example of BMN

  To help the users have a basic idea of a complete config structure and the modules in an action localization system,
  we make brief comments on the config of BMN as the following.
  For more detailed usage and alternative for per parameter in each module, please refer to the [API documentation](https://mmaction2.readthedocs.io/en/latest/api.html).

  ```python
  # model settings
  model = dict(  # Config of the model
      type='BMN',  # Class name of the localizer
      temporal_dim=100,  # Total frames selected for each video
      boundary_ratio=0.5,  # Ratio for determining video boundaries
      num_samples=32,  # Number of samples for each proposal
      num_samples_per_bin=3,  # Number of bin samples for each sample
      feat_dim=400,  # Dimension of feature
      soft_nms_alpha=0.4,  # Soft NMS alpha
      soft_nms_low_threshold=0.5,  # Soft NMS low threshold
      soft_nms_high_threshold=0.9,  # Soft NMS high threshold
      post_process_top_k=100)  # Top k proposals in post process

  # dataset settings
  dataset_type = 'ActivityNetDataset'  # Type of dataset for training, validation and testing
  data_root = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for training
  data_root_val = 'data/activitynet_feature_cuhk/csv_mean_100/'  # Root path to data for validation and testing
  ann_file_train = 'data/ActivityNet/anet_anno_train.json'  # Path to the annotation file for training
  ann_file_val = 'data/ActivityNet/anet_anno_val.json'  # Path to the annotation file for validation
  ann_file_test = 'data/ActivityNet/anet_anno_test.json'  # Path to the annotation file for testing

  train_pipeline = [  # Training data processing pipeline
      dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
      dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
      dict(
          type='PackLocalizationInputs', # Pack localization data
          keys=('gt_bbox'), # Keys of input
          meta_keys=('video_name'))] # Meta keys of input
  val_pipeline = [  # Validation data processing pipeline
      dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
      dict(type='GenerateLocalizationLabels'),  # Generate localization labels pipeline
      dict(
          type='PackLocalizationInputs',  # Pack localization data
          keys=('gt_bbox'),   # Keys of input
          meta_keys=('video_name', 'duration_second', 'duration_frame',
                     'annotations', 'feature_frame'))]  # Meta keys of input
  test_pipeline = [  # Testing data processing pipeline
      dict(type='LoadLocalizationFeature'),  # Load localization feature pipeline
      dict(
          type='PackLocalizationInputs',  # Pack localization data
          keys=('gt_bbox'),  # Keys of input
          meta_keys=('video_name', 'duration_second', 'duration_frame',
                     'annotations', 'feature_frame'))]  # Meta keys of input
  train_dataloader = dict(  # Config of train dataloader
      batch_size=8,  # Batch size of each single GPU during training
      num_workers=8,  # Workers to pre-fetch data for each single GPU during training
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end, which can accelerate training speed
      sampler=dict(
          type='DefaultSampler',  # DefaultSampler which supports both distributed and non-distributed training. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/dataset/sampler.py
          shuffle=True),  # Randomly shuffle the training data in each epoch
      dataset=dict(  # Config of train dataset
          type=dataset_type,
          ann_file=ann_file_train,  # Path of annotation file
          data_prefix=dict(video=data_root),  # Prefix of video path
          pipeline=train_pipeline))
  val_dataloader = dict(  # Config of validation dataloader
      batch_size=1,  # Batch size of each single GPU during evaluation
      num_workers=8,  # Workers to pre-fetch data for each single GPU during evaluation
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # Not shuffle during validation and testing
      dataset=dict(  # Config of validation dataset
          type=dataset_type,
          ann_file=ann_file_val,  # Path of annotation file
          data_prefix=dict(video=data_root_val),  # Prefix of video path
          pipeline=val_pipeline,
          test_mode=True))
  test_dataloader = dict(  # Config of test dataloader
      batch_size=1,  # Batch size of each single GPU during testing
      num_workers=8,  # Workers to pre-fetch data for each single GPU during testing
      persistent_workers=True,  # If `True`, the dataloader will not shut down the worker processes after an epoch end
      sampler=dict(
          type='DefaultSampler',
          shuffle=False),  # Not shuffle during validation and testing
      dataset=dict(  # Config of test dataset
          type=dataset_type,
          ann_file=ann_file_val,  # Path of annotation file
          data_prefix=dict(video=data_root_val),  # Prefix of video path
          pipeline=test_pipeline,
          test_mode=True))

  # evaluation settings
  work_dir = './work_dirs/bmn_400x100_2x8_9e_activitynet_feature/'  # Directory to save the model checkpoints and logs for the current experiments
  val_evaluator = dict(
    type='ANetMetric',
    metric_type='AR@AN',
    dump_config=dict(  # Config of localization output
        out=f'{work_dir}/results.json',  # Path to the output file
        output_format='json'))  # File format of the output file
  test_evaluator = val_evaluator   # Set test_evaluator as val_evaluator

  max_epochs = 9  # Total epochs to train the model
  train_cfg = dict(  # Config of training loop
      type='EpochBasedTrainLoop',  # Name of training loop
      max_epochs=max_epochs,  # Total training epochs
      val_begin=1,  # The epoch that begins validating
      val_interval=1)  # Validation interval
  val_cfg = dict(  # Config of validation loop
      type='ValLoop')  # Name of validating loop
  test_cfg = dict( # Config of testing loop
      type='TestLoop')  # Name of testing loop

  # learning policy
  param_scheduler = [  # Parameter scheduler for updating optimizer parameters, support dict or list
      dict(type='MultiStepLR',  # Decays the learning rate once the number of epoch reaches one of the milestones
      begin=0,  # Step at which to start updating the learning rate
      end=max_epochs,  # Step at which to stop updating the learning rate
      by_epoch=True,  # Whether the scheduled learning rate is updated by epochs
      milestones=[7, ],  # Steps to decay the learning rate
      gamma=0.1)]  # Multiplicative factor of parameter value decay

  # optimizer
  optim_wrapper = dict(  # Config of optimizer wrapper
      type='OptimWrapper',  # Name of optimizer wrapper, switch to AmpOptimWrapper to enable mixed precision training
      optimizer=dict(  # Config of optimizer. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
          type='Adam',  # Name of optimizer
          lr=0.001,  # Learning rate
          weight_decay=0.0001),  # Weight decay
      clip_grad=dict(max_norm=40, norm_type=2))  # Config of gradient clip

  # runtime settings
  default_scope = 'mmaction'  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/tutorials/registry.html
  default_hooks = dict(  # Hooks to execute default actions like updating model parameters and saving checkpoints.
      runtime_info=dict(type='RuntimeInfoHook'),  # The hook to updates runtime information into message hub
      timer=dict(type='IterTimerHook'),  # The logger used to record time spent during iteration
      logger=dict(
          type='LoggerHook',  # The logger used to record logs during training/validation/testing phase
          interval=20,  # Interval to print the log
          ignore_last=False), # Ignore the log of last iterations in each epoch
      param_scheduler=dict(type='ParamSchedulerHook'),  # The hook to update some hyper-parameters in optimizer
      checkpoint=dict(
          type='CheckpointHook',  # The hook to save checkpoints periodically
          interval=3,  # The saving period
          save_best='auto',  # Specified metric to mearsure the best checkpoint during evaluation
          max_keep_ckpts=3),  # The maximum checkpoints to keep
      sampler_seed=dict(type='DistSamplerSeedHook'),  # Data-loading sampler for distributed training
      sync_buffers=dict(type='SyncBuffersHook'))  # Synchronize model buffers at the end of each epoch
  env_cfg = dict(  # Dict for setting environment
      cudnn_benchmark=False,  # Whether to enable cudnn benchmark
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0), # Parameters to setup multiprocessing
      dist_cfg=dict(backend='nccl')) # Parameters to setup distributed environment, the port can also be set

  log_processor = dict(
      type='LogProcessor',  # Log processor used to format log information
      window_size=20,  # Default smooth interval
      by_epoch=True)  # Whether to format logs with epoch type
  vis_backends = [  # List of visualization backends
      dict(type='LocalVisBackend')]  # Local visualization backend
  visualizer = dict(  # Config of visualizer
      type='ActionVisualizer',  # Name of visualizer
      vis_backends=vis_backends)
  log_level = 'INFO'  # The level of logging
  load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
  resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
  ```
