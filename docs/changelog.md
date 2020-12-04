## Changelog

### 0.10.0 (31/12/2020)

**Highlights**

**New Features**

**Improvements**
- Add FAQ documents for easy troubleshooting. ([#413](https://github.com/open-mmlab/mmaction2/pull/413), [#420](https://github.com/open-mmlab/mmaction2/pull/420))


**Bug and Typo Fixes**

**ModelZoo**

### 0.9.0 (30/11/2020)

**Highlights**
- Support GradCAM utils for recognizers
- Support ResNet Audio model

**New Features**
- Automatically add modelzoo statistics to readthedocs ([#327](https://github.com/open-mmlab/mmaction2/pull/327))
- Support GYM99 data preparation ([#331](https://github.com/open-mmlab/mmaction2/pull/331))
- Add AudioOnly Pathway from AVSlowFast. ([#355](https://github.com/open-mmlab/mmaction2/pull/355))
- Add GradCAM utils for recognizer ([#324](https://github.com/open-mmlab/mmaction2/pull/324))
- Add print config script ([#345](https://github.com/open-mmlab/mmaction2/pull/345))
- Add online motion vector decoder ([#291](https://github.com/open-mmlab/mmaction2/pull/291))

**Improvements**
- Support PyTorch 1.7 in CI ([#312](https://github.com/open-mmlab/mmaction2/pull/312))
- Support to predict different labels in a long video ([#274](https://github.com/open-mmlab/mmaction2/pull/274))
- Update docs bout test crops ([#359](https://github.com/open-mmlab/mmaction2/pull/359))
- Polish code format using pylint manually ([#338](https://github.com/open-mmlab/mmaction2/pull/338))
- Update unittest coverage ([#358](https://github.com/open-mmlab/mmaction2/pull/358), [#322](https://github.com/open-mmlab/mmaction2/pull/322), [#325](https://github.com/open-mmlab/mmaction2/pull/325))
- Add random seed for building filelists ([#323](https://github.com/open-mmlab/mmaction2/pull/323))
- Update colab tutorial ([#367](https://github.com/open-mmlab/mmaction2/pull/367))
- set default batch_size of evaluation and testing to 1 ([#250](https://github.com/open-mmlab/mmaction2/pull/250))
- Rename the preparation docs to `README.md` ([#388](https://github.com/open-mmlab/mmaction2/pull/388))
- Move docs about demo to `demo/README.md` ([#329](https://github.com/open-mmlab/mmaction2/pull/329))
- Remove redundant code in `tools/test.py` ([#310](https://github.com/open-mmlab/mmaction2/pull/310))
- Automatically calculate number of test clips for Recognizer2D ([#359](https://github.com/open-mmlab/mmaction2/pull/359))

**Bug and Typo Fixes**
- Fix rename Kinetics classnames bug ([#384](https://github.com/open-mmlab/mmaction2/pull/384))
- Fix a bug in BaseDataset when `data_prefix` is None ([#314](https://github.com/open-mmlab/mmaction2/pull/314))
- Fix a bug about `tmp_folder` in `OpenCVInit` ([#357](https://github.com/open-mmlab/mmaction2/pull/357))
- Fix `get_thread_id` when not using disk as backend ([#354](https://github.com/open-mmlab/mmaction2/pull/354), [#357](https://github.com/open-mmlab/mmaction2/pull/357))
- Fix the bug of HVU object `num_classes` from 1679 to 1678 ([#307](https://github.com/open-mmlab/mmaction2/pull/307))
- Fix typo in `export_model.md` ([#399](https://github.com/open-mmlab/mmaction2/pull/399))
- Fix OmniSource training configs ([#321](https://github.com/open-mmlab/mmaction2/pull/321))
- Fix Issue #306: Bug of SampleAVAFrames ([#317](https://github.com/open-mmlab/mmaction2/pull/317))

**ModelZoo**
- Add SlowOnly model for GYM99, both RGB and Flow ([#336](https://github.com/open-mmlab/mmaction2/pull/336))
- Add auto modelzoo statistics in readthedocs ([#327](https://github.com/open-mmlab/mmaction2/pull/327))
- Add TSN for HMDB51 pretrained on Kinetics400, Moments in Time and ImageNet. ([#372](https://github.com/open-mmlab/mmaction2/pull/372))

### v0.8.0 (31/10/2020)

**Highlights**
- Support [OmniSource](https://arxiv.org/abs/2003.13042)
- Support C3D
- Support video recognition with audio modality
- Support HVU
- Support X3D

**New Features**
- Support AVA dataset preparation ([#266](https://github.com/open-mmlab/mmaction2/pull/266))
- Support the training of video recognition dataset with multiple tag categories ([#235](https://github.com/open-mmlab/mmaction2/pull/235))
- Support joint training with multiple training datasets of multiple formats, including images, untrimmed videos, etc. ([#242](https://github.com/open-mmlab/mmaction2/pull/242))
- Support to specify a start epoch to conduct evaluation ([#216](https://github.com/open-mmlab/mmaction2/pull/216))
- Implement X3D models, support testing with model weights converted from SlowFast ([#288](https://github.com/open-mmlab/mmaction2/pull/288))
- Support specify a start epoch to conduct evaluation ([#216](https://github.com/open-mmlab/mmaction2/pull/216))

**Improvements**
- Set default values of 'average_clips' in each config file so that there is no need to set it explicitly during testing in most cases ([#232](https://github.com/open-mmlab/mmaction2/pull/232))
- Extend HVU datatools to generate individual file list for each tag category ([#258](https://github.com/open-mmlab/mmaction2/pull/258))
- Support data preparation for Kinetics-600 and Kinetics-700 ([#254](https://github.com/open-mmlab/mmaction2/pull/254))
- Use `metric_dict` to replace hardcoded arguments in `evaluate` function ([#286](https://github.com/open-mmlab/mmaction2/pull/286))
- Add `cfg-options` in arguments to override some settings in the used config for convenience ([#212](https://github.com/open-mmlab/mmaction2/pull/212))
- Rename the old evaluating protocol `mean_average_precision` as `mmit_mean_average_precision` since it is only used on MMIT and is not the `mAP` we usually talk about. Add `mean_average_precision`, which is the real `mAP` ([#235](https://github.com/open-mmlab/mmaction2/pull/235))
- Add accurate setting (Three crop * 2 clip) and report corresponding performance for TSM model ([#241](https://github.com/open-mmlab/mmaction2/pull/241))
- Add citations in each preparing_dataset.md in `tools/data/dataset` ([#289](https://github.com/open-mmlab/mmaction2/pull/289))
- Update the performance of audio-visual fusion on Kinetics-400 ([#281](https://github.com/open-mmlab/mmaction2/pull/281))
- Support data preparation of OmniSource web datasets, including GoogleImage, InsImage, InsVideo and KineticsRawVideo ([#294](https://github.com/open-mmlab/mmaction2/pull/294))
- Use `metric_options` dict to provide metric args in `evaluate` ([#286](https://github.com/open-mmlab/mmaction2/pull/286))

**Bug Fixes**
- Register `FrameSelector` in `PIPELINES` ([#268](https://github.com/open-mmlab/mmaction2/pull/268))
- Fix the potential bug for default value in dataset_setting ([#245](https://github.com/open-mmlab/mmaction2/pull/245))
- Fix multi-node dist test ([#292](https://github.com/open-mmlab/mmaction2/pull/292))
- Fix the data preparation bug for `something-something` dataset ([#278](https://github.com/open-mmlab/mmaction2/pull/278))
- Fix the invalid config url in slowonly README data benchmark ([#249](https://github.com/open-mmlab/mmaction2/pull/249))
- Validate that the performance of models trained with videos have no significant difference comparing to the performance of models trained with rawframes ([#256](https://github.com/open-mmlab/mmaction2/pull/256))
- Correct the `img_norm_cfg` used by TSN-3seg-R50 UCF-101 model, improve the Top-1 accuracy by 3% ([#273](https://github.com/open-mmlab/mmaction2/pull/273))

**ModelZoo**
- Add Baselines for Kinetics-600 and Kinetics-700, including TSN-R50-8seg and SlowOnly-R50-8x8 ([#259](https://github.com/open-mmlab/mmaction2/pull/259))
- Add OmniSource benchmark on MiniKineitcs ([#296](https://github.com/open-mmlab/mmaction2/pull/296))
- Add Baselines for HVU, including TSN-R18-8seg on 6 tag categories of HVU ([#287](https://github.com/open-mmlab/mmaction2/pull/287))
- Add X3D models ported from [SlowFast](https://github.com/facebookresearch/SlowFast/) ([#288](https://github.com/open-mmlab/mmaction2/pull/288))

### v0.7.0 (30/9/2020)

**Highlights**
- Support TPN
- Support JHMDB, UCF101-24, HVU dataset preparation
- support onnx model conversion

**New Features**
- Support the data pre-processing pipeline for the HVU Dataset ([#277](https://github.com/open-mmlab/mmaction2/pull/227/))
- Support real-time action recognition from web camera ([#171](https://github.com/open-mmlab/mmaction2/pull/171))
- Support onnx ([#160](https://github.com/open-mmlab/mmaction2/pull/160))
- Support UCF101-24 preparation ([#219](https://github.com/open-mmlab/mmaction2/pull/219))
- Support evaluating mAP for ActivityNet with [CUHK17_activitynet_pred](http://activity-net.org/challenges/2017/evaluation.html) ([#176](https://github.com/open-mmlab/mmaction2/pull/176))
- Add the data pipeline for ActivityNet, including downloading videos, extracting RGB and Flow frames, finetuning TSN and extracting feature ([#190](https://github.com/open-mmlab/mmaction2/pull/190))
- Support JHMDB preparation ([#220](https://github.com/open-mmlab/mmaction2/pull/220))

**ModelZoo**
- Add finetuning setting for SlowOnly ([#173](https://github.com/open-mmlab/mmaction2/pull/173))
- Add TSN and SlowOnly models trained with [OmniSource](https://arxiv.org/abs/2003.13042), which achieve 75.7% Top-1 with TSN-R50-3seg and 80.4% Top-1 with SlowOnly-R101-8x8 ([#215](https://github.com/open-mmlab/mmaction2/pull/215))

**Improvements**
- Support demo with video url ([#165](https://github.com/open-mmlab/mmaction2/pull/165))
- Support multi-batch when testing ([#184](https://github.com/open-mmlab/mmaction2/pull/184))
- Add tutorial for adding a new learning rate updater ([#181](https://github.com/open-mmlab/mmaction2/pull/181))
- Add config name in meta info ([#183](https://github.com/open-mmlab/mmaction2/pull/183))
- Remove git hash in `__version__` ([#189](https://github.com/open-mmlab/mmaction2/pull/189))
- Check mmcv version ([#189](https://github.com/open-mmlab/mmaction2/pull/189))
- Update url with 'https://download.openmmlab.com' ([#208](https://github.com/open-mmlab/mmaction2/pull/208))
- Update Docker file to support PyTorch 1.6 and update `install.md` ([#209](https://github.com/open-mmlab/mmaction2/pull/209))
- Polish readsthedocs display ([#217](https://github.com/open-mmlab/mmaction2/pull/217), [#229](https://github.com/open-mmlab/mmaction2/pull/229))

**Bug Fixes**
- Fix the bug when using OpenCV to extract only RGB frames with original shape ([#184](https://github.com/open-mmlab/mmaction2/pull/187))
- Fix the bug of sthv2 `num_classes` from 339 to 174 ([#174](https://github.com/open-mmlab/mmaction2/pull/174), [#207](https://github.com/open-mmlab/mmaction2/pull/207))


### v0.6.0 (2/9/2020)

**Highlights**
- Support TIN, CSN, SSN, NonLocal
- Support FP16 training

**New Features**
- Support NonLocal module and provide ckpt in TSM and I3D ([#41](https://github.com/open-mmlab/mmaction2/pull/41))
- Support SSN ([#33](https://github.com/open-mmlab/mmaction2/pull/33), [#37](https://github.com/open-mmlab/mmaction2/pull/37), [#52](https://github.com/open-mmlab/mmaction2/pull/52), [#55](https://github.com/open-mmlab/mmaction2/pull/55))
- Support CSN ([#87](https://github.com/open-mmlab/mmaction2/pull/87))
- Support TIN ([#53](https://github.com/open-mmlab/mmaction2/pull/53))
- Support HMDB51 dataset preparation ([#60](https://github.com/open-mmlab/mmaction2/pull/60))
- Support encoding videos from frames ([#84](https://github.com/open-mmlab/mmaction2/pull/84))
- Support FP16 training ([#25](https://github.com/open-mmlab/mmaction2/pull/25))
- Enhance demo by supporting rawframe inference ([#59](https://github.com/open-mmlab/mmaction2/pull/59)), output video/gif ([#72](https://github.com/open-mmlab/mmaction2/pull/72))

**ModelZoo**
- Update Slowfast modelzoo ([#51](https://github.com/open-mmlab/mmaction2/pull/51))
- Update TSN, TSM video checkpoints ([#50](https://github.com/open-mmlab/mmaction2/pull/50))
- Add data benchmark for TSN ([#57](https://github.com/open-mmlab/mmaction2/pull/57))
- Add data benchmark for SlowOnly ([#77](https://github.com/open-mmlab/mmaction2/pull/77))
- Add BSN/BMN performance results with feature extracted by our codebase ([#99](https://github.com/open-mmlab/mmaction2/pull/99))

**Improvements**
- Polish data preparation codes ([#70](https://github.com/open-mmlab/mmaction2/pull/70))
- Improve data preparation scripts ([#58](https://github.com/open-mmlab/mmaction2/pull/58))
- Improve unittest coverage and minor fix ([#62](https://github.com/open-mmlab/mmaction2/pull/62))
- Support PyTorch 1.6 in CI ([#117](https://github.com/open-mmlab/mmaction2/pull/117))
- Support `with_offset` for rawframe dataset ([#48](https://github.com/open-mmlab/mmaction2/pull/48))
- Support json annotation files ([#119](https://github.com/open-mmlab/mmaction2/pull/119))
- Support `multi-class` in TSMHead ([#104](https://github.com/open-mmlab/mmaction2/pull/104))
- Support using `val_step()` to validate data for each `val` workflow ([#123](https://github.com/open-mmlab/mmaction2/pull/123))
- Use `xxInit()` method to get `total_frames` and make `total_frames` a required key ([#90](https://github.com/open-mmlab/mmaction2/pull/90))
- Add paper introduction in model readme ([#140](https://github.com/open-mmlab/mmaction2/pull/140))
- Adjust the directory structure of `tools/` and rename some scripts files ([#142](https://github.com/open-mmlab/mmaction2/pull/142))

**Bug Fixes**
- Fix configs for localization test ([#67](https://github.com/open-mmlab/mmaction2/pull/67))
- Fix configs of SlowOnly by fixing lr to 8 gpus ([#136](https://github.com/open-mmlab/mmaction2/pull/136))
- Fix the bug in analyze_log ([#54](https://github.com/open-mmlab/mmaction2/pull/54))
- Fix the bug of generating HMDB51 class index file ([#69](https://github.com/open-mmlab/mmaction2/pull/69))
- Fix the bug of using `load_checkpoint()` in ResNet ([#93](https://github.com/open-mmlab/mmaction2/pull/93))
- Fix the bug of `--work-dir` when using slurm training script ([#110](https://github.com/open-mmlab/mmaction2/pull/110))
- Correct the sthv1/sthv2 rawframes filelist generate command ([#71](https://github.com/open-mmlab/mmaction2/pull/71))
- `CosineAnnealing` typo ([#47](https://github.com/open-mmlab/mmaction2/pull/47))


### v0.5.0 (9/7/2020)

**Highlights**
- MMAction2 is released

**New Features**
- Support various datasets: UCF101, Kinetics-400, Something-Something V1&V2, Moments in Time,
  Multi-Moments in Time, THUMOS14
- Support various action recognition methods: TSN, TSM, R(2+1)D, I3D, SlowOnly, SlowFast, Non-local
- Support various action localization methods: BSN, BMN
- Colab demo for action recognition
