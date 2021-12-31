## Changelog

### 0.21.0 (31/12/2021)

**Highlights**

- Support 2s-AGCN
- Support publish models in Windows
- Improve some sthv1 related models
- Support BABEL

**New Features**

- Support 2s-AGCN([#1248](https://github.com/open-mmlab/mmaction2/pull/1248))
- Support skip postproc in ntu_pose_extraction([#1295](https://github.com/open-mmlab/mmaction2/pull/1295))
- Support publish models in Windows([#1325](https://github.com/open-mmlab/mmaction2/pull/1325))
- Add copyright checkhook in pre-commit-config([#1344](https://github.com/open-mmlab/mmaction2/pull/1344))

**Documentations**

- Add MMFlow ([#1273](https://github.com/open-mmlab/mmaction2/pull/1273))
- Revise README.md and add projects.md ([#1286](https://github.com/open-mmlab/mmaction2/pull/1286))
- Add 2s-AGCN in Updates([#1289](https://github.com/open-mmlab/mmaction2/pull/1289))
- Add MMFewShot([#1300](https://github.com/open-mmlab/mmaction2/pull/1300))
- Add MMHuman3d([#1304](https://github.com/open-mmlab/mmaction2/pull/1304))
- Update pre-commit([#1313](https://github.com/open-mmlab/mmaction2/pull/1313))
- Use share menu from the theme instead([#1328](https://github.com/open-mmlab/mmaction2/pull/1328))
- Update installation command([#1340](https://github.com/open-mmlab/mmaction2/pull/1340))

**Bug and Typo Fixes**

- Update the inference part in notebooks([#1256](https://github.com/open-mmlab/mmaction2/pull/1256))
- Update the map_location([#1262]((https://github.com/open-mmlab/mmaction2/pull/1262)))
- Fix bug that start_index is not used in RawFrameDecode([#1278](https://github.com/open-mmlab/mmaction2/pull/1278))
- Fix bug in init_random_seed([#1282](https://github.com/open-mmlab/mmaction2/pull/1282))
- Fix bug in setup.py([#1303](https://github.com/open-mmlab/mmaction2/pull/1303))
- Fix interrogate error in workflows([#1305](https://github.com/open-mmlab/mmaction2/pull/1305))
- Fix typo in slowfast config([#1309](https://github.com/open-mmlab/mmaction2/pull/1309))
- Cancel previous runs that are not completed([#1327](https://github.com/open-mmlab/mmaction2/pull/1327))
- Fix missing skip_postproc parameter([#1347](https://github.com/open-mmlab/mmaction2/pull/1347))
- Update ssn.py([#1355](https://github.com/open-mmlab/mmaction2/pull/1355))
- Use latest youtube-dl([#1357](https://github.com/open-mmlab/mmaction2/pull/1357))
- Fix test-best([#1362](https://github.com/open-mmlab/mmaction2/pull/1362))

**ModelZoo**

- Improve some sthv1 related models([#1306](https://github.com/open-mmlab/mmaction2/pull/1306))
- Support BABEL([#1332](https://github.com/open-mmlab/mmaction2/pull/1332))

### 0.20.0 (07/10/2021)

**Highlights**

- Support TorchServe
- Add video structuralize demo
- Support using 3D skeletons for skeleton-based action recognition
- Benchmark PoseC3D on UCF and HMDB

**New Features**

- Support TorchServe ([#1212](https://github.com/open-mmlab/mmaction2/pull/1212))
- Support 3D skeletons pre-processing ([#1218](https://github.com/open-mmlab/mmaction2/pull/1218))
- Support video structuralize demo ([#1197](https://github.com/open-mmlab/mmaction2/pull/1197))

**Documentations**

- Revise README.md and add projects.md ([#1214](https://github.com/open-mmlab/mmaction2/pull/1214))
- Add CN docs for Skeleton dataset, PoseC3D and ST-GCN ([#1228](https://github.com/open-mmlab/mmaction2/pull/1228), [#1237](https://github.com/open-mmlab/mmaction2/pull/1237), [#1236](https://github.com/open-mmlab/mmaction2/pull/1236))
- Add tutorial for custom dataset training for skeleton-based action recognition ([#1234](https://github.com/open-mmlab/mmaction2/pull/1234))

**Bug and Typo Fixes**

- Fix tutorial link ([#1219](https://github.com/open-mmlab/mmaction2/pull/1219))
- Fix GYM links ([#1224](https://github.com/open-mmlab/mmaction2/pull/1224))

**ModelZoo**

- Benchmark PoseC3D on UCF and HMDB ([#1223](https://github.com/open-mmlab/mmaction2/pull/1223))
- Add ST-GCN + 3D skeleton model for NTU60-XSub ([#1236](https://github.com/open-mmlab/mmaction2/pull/1236))

### 0.19.0 (07/10/2021)

**Highlights**

- Support ST-GCN
- Refactor the inference API
- Add code spell check hook

**New Features**

- Support ST-GCN ([#1123](https://github.com/open-mmlab/mmaction2/pull/1123))

**Improvement**

- Add label maps for every dataset ([#1127](https://github.com/open-mmlab/mmaction2/pull/1127))
- Remove useless code MultiGroupCrop ([#1180](https://github.com/open-mmlab/mmaction2/pull/1180))
- Refactor Inference API ([#1191](https://github.com/open-mmlab/mmaction2/pull/1191))
- Add code spell check hook ([#1208](https://github.com/open-mmlab/mmaction2/pull/1208))
- Use docker in CI ([#1159](https://github.com/open-mmlab/mmaction2/pull/1159))

**Documentations**

- Update metafiles to new OpenMMLAB protocols ([#1134](https://github.com/open-mmlab/mmaction2/pull/1134))
- Switch to new doc style ([#1160](https://github.com/open-mmlab/mmaction2/pull/1160))
- Improve the ERROR message ([#1203](https://github.com/open-mmlab/mmaction2/pull/1203))
- Fix invalid URL in getting_started ([#1169](https://github.com/open-mmlab/mmaction2/pull/1169))

**Bug and Typo Fixes**

- Compatible with new MMClassification ([#1139](https://github.com/open-mmlab/mmaction2/pull/1139))
- Add missing runtime dependencies ([#1144](https://github.com/open-mmlab/mmaction2/pull/1144))
- Fix THUMOS tag proposals path ([#1156](https://github.com/open-mmlab/mmaction2/pull/1156))
- Fix LoadHVULabel ([#1194](https://github.com/open-mmlab/mmaction2/pull/1194))
- Switch the default value of `persistent_workers` to False ([#1202](https://github.com/open-mmlab/mmaction2/pull/1202))
- Fix `_freeze_stages` for MobileNetV2 ([#1193](https://github.com/open-mmlab/mmaction2/pull/1193))
- Fix resume when building rawframes ([#1150](https://github.com/open-mmlab/mmaction2/pull/1150))
- Fix device bug for class weight ([#1188](https://github.com/open-mmlab/mmaction2/pull/1188))
- Correct Arg names in extract_audio.py ([#1148](https://github.com/open-mmlab/mmaction2/pull/1148))

**ModelZoo**

- Add TSM-MobileNetV2 ported from TSM ([#1163](https://github.com/open-mmlab/mmaction2/pull/1163))
- Add ST-GCN for NTURGB+D-XSub-60 ([#1123](https://github.com/open-mmlab/mmaction2/pull/1123))

### 0.18.0 (02/09/2021)

**Improvement**

- Add CopyRight ([#1099](https://github.com/open-mmlab/mmaction2/pull/1099))
- Support NTU Pose Extraction ([#1076](https://github.com/open-mmlab/mmaction2/pull/1076))
- Support Caching in RawFrameDecode ([#1078](https://github.com/open-mmlab/mmaction2/pull/1078))
- Add citations & Support python3.9 CI & Use fixed-version sphinx ([#1125](https://github.com/open-mmlab/mmaction2/pull/1125))

**Documentations**

- Add Descriptions of PoseC3D dataset ([#1053](https://github.com/open-mmlab/mmaction2/pull/1053))

**Bug and Typo Fixes**

- Fix SSV2 checkpoints ([#1101](https://github.com/open-mmlab/mmaction2/pull/1101))
- Fix CSN normalization ([#1116](https://github.com/open-mmlab/mmaction2/pull/1116))
- Fix typo ([#1121](https://github.com/open-mmlab/mmaction2/pull/1121))
- Fix new_crop_quadruple bug ([#1108](https://github.com/open-mmlab/mmaction2/pull/1108))

### 0.17.0 (03/08/2021)

**Highlights**

- Support PyTorch 1.9
- Support Pytorchvideo Transforms
- Support PreciseBN

**New Features**

- Support Pytorchvideo Transforms ([#1008](https://github.com/open-mmlab/mmaction2/pull/1008))
- Support PreciseBN ([#1038](https://github.com/open-mmlab/mmaction2/pull/1038))

**Improvements**

- Remove redundant augmentations in config files ([#996](https://github.com/open-mmlab/mmaction2/pull/996))
- Make resource directory to hold common resource pictures ([#1011](https://github.com/open-mmlab/mmaction2/pull/1011))
- Remove deprecated FrameSelector ([#1010](https://github.com/open-mmlab/mmaction2/pull/1010))
- Support Concat Dataset ([#1000](https://github.com/open-mmlab/mmaction2/pull/1000))
- Add `to-mp4` option to resize_videos.py ([#1021](https://github.com/open-mmlab/mmaction2/pull/1021))
- Add option to keep tail frames ([#1050](https://github.com/open-mmlab/mmaction2/pull/1050))
- Update MIM support ([#1061](https://github.com/open-mmlab/mmaction2/pull/1061))
- Calculate Top-K accurate and inaccurate classes ([#1047](https://github.com/open-mmlab/mmaction2/pull/1047))

**Bug and Typo Fixes**

- Fix bug in PoseC3D demo ([#1009](https://github.com/open-mmlab/mmaction2/pull/1009))
- Fix some problems in resize_videos.py ([#1012](https://github.com/open-mmlab/mmaction2/pull/1012))
- Support torch1.9 ([#1015](https://github.com/open-mmlab/mmaction2/pull/1015))
- Remove redundant code in CI ([#1046](https://github.com/open-mmlab/mmaction2/pull/1046))
- Fix bug about persistent_workers ([#1044](https://github.com/open-mmlab/mmaction2/pull/1044))
- Support TimeSformer feature extraction ([#1035](https://github.com/open-mmlab/mmaction2/pull/1035))
- Fix ColorJitter ([#1025](https://github.com/open-mmlab/mmaction2/pull/1025))

**ModelZoo**

- Add TSM-R50 sthv1 models trained by PytorchVideo RandAugment and AugMix ([#1008](https://github.com/open-mmlab/mmaction2/pull/1008))
- Update SlowOnly SthV1 checkpoints ([#1034](https://github.com/open-mmlab/mmaction2/pull/1034))
- Add SlowOnly Kinetics400 checkpoints trained with Precise-BN ([#1038](https://github.com/open-mmlab/mmaction2/pull/1038))
- Add CSN-R50 from scratch checkpoints ([#1045](https://github.com/open-mmlab/mmaction2/pull/1045))
- TPN Kinetics-400 Checkpoints trained with the new ColorJitter ([#1025](https://github.com/open-mmlab/mmaction2/pull/1025))

**Documentation**

- Add Chinese translation of feature_extraction.md ([#1020](https://github.com/open-mmlab/mmaction2/pull/1020))
- Fix the code snippet in getting_started.md ([#1023](https://github.com/open-mmlab/mmaction2/pull/1023))
- Fix TANet config table ([#1028](https://github.com/open-mmlab/mmaction2/pull/1028))
- Add description to PoseC3D dataset ([#1053](https://github.com/open-mmlab/mmaction2/pull/1053))

### 0.16.0 (01/07/2021)

**Highlights**

- Support using backbone from pytorch-image-models(timm)
- Support PIMS Decoder
- Demo for skeleton-based action recognition
- Support Timesformer

**New Features**

- Support using backbones from pytorch-image-models(timm) for TSN ([#880](https://github.com/open-mmlab/mmaction2/pull/880))
- Support torchvision transformations in preprocessing pipelines ([#972](https://github.com/open-mmlab/mmaction2/pull/972))
- Demo for skeleton-based action recognition ([#972](https://github.com/open-mmlab/mmaction2/pull/972))
- Support Timesformer ([#839](https://github.com/open-mmlab/mmaction2/pull/839))

**Improvements**

- Add a tool to find invalid videos ([#907](https://github.com/open-mmlab/mmaction2/pull/907), [#950](https://github.com/open-mmlab/mmaction2/pull/950))
- Add an option to specify spectrogram_type ([#909](https://github.com/open-mmlab/mmaction2/pull/909))
- Add json output to video demo ([#906](https://github.com/open-mmlab/mmaction2/pull/906))
- Add MIM related docs ([#918](https://github.com/open-mmlab/mmaction2/pull/918))
- Rename lr to scheduler ([#916](https://github.com/open-mmlab/mmaction2/pull/916))
- Support `--cfg-options` for demos ([#911](https://github.com/open-mmlab/mmaction2/pull/911))
- Support number counting for flow-wise filename template ([#922](https://github.com/open-mmlab/mmaction2/pull/922))
- Add Chinese tutorial ([#941](https://github.com/open-mmlab/mmaction2/pull/941))
- Change ResNet3D default values ([#939](https://github.com/open-mmlab/mmaction2/pull/939))
- Adjust script structure ([#935](https://github.com/open-mmlab/mmaction2/pull/935))
- Add font color to args in long_video_demo ([#947](https://github.com/open-mmlab/mmaction2/pull/947))
- Polish code style with Pylint ([#908](https://github.com/open-mmlab/mmaction2/pull/908))
- Support PIMS Decoder ([#946](https://github.com/open-mmlab/mmaction2/pull/946))
- Improve Metafiles ([#956](https://github.com/open-mmlab/mmaction2/pull/956), [#979](https://github.com/open-mmlab/mmaction2/pull/979), [#966](https://github.com/open-mmlab/mmaction2/pull/966))
- Add links to download Kinetics400 validation ([#920](https://github.com/open-mmlab/mmaction2/pull/920))
- Audit the usage of shutil.rmtree ([#943](https://github.com/open-mmlab/mmaction2/pull/943))
- Polish localizer related codes([#913](https://github.com/open-mmlab/mmaction2/pull/913))

**Bug and Typo Fixes**

- Fix spatiotemporal detection demo ([#899](https://github.com/open-mmlab/mmaction2/pull/899))
- Fix docstring for 3D inflate ([#925](https://github.com/open-mmlab/mmaction2/pull/925))
- Fix bug of writing text to video with TextClip ([#952](https://github.com/open-mmlab/mmaction2/pull/952))
- Fix mmcv install in CI ([#977](https://github.com/open-mmlab/mmaction2/pull/977))

**ModelZoo**

- Add TSN with Swin Transformer backbone as an example for using pytorch-image-models(timm) backbones ([#880](https://github.com/open-mmlab/mmaction2/pull/880))
- Port CSN checkpoints from VMZ ([#945](https://github.com/open-mmlab/mmaction2/pull/945))
- Release various checkpoints for UCF101, HMDB51 and Sthv1 ([#938](https://github.com/open-mmlab/mmaction2/pull/938))
- Support Timesformer ([#839](https://github.com/open-mmlab/mmaction2/pull/839))
- Update TSM modelzoo ([#981](https://github.com/open-mmlab/mmaction2/pull/981))

### 0.15.0 (31/05/2021)

**Highlights**

- Support PoseC3D
- Support ACRN
- Support MIM

**New Features**

- Support PoseC3D ([#786](https://github.com/open-mmlab/mmaction2/pull/786), [#890](https://github.com/open-mmlab/mmaction2/pull/890))
- Support MIM ([#870](https://github.com/open-mmlab/mmaction2/pull/870))
- Support ACRN and Focal Loss ([#891](https://github.com/open-mmlab/mmaction2/pull/891))
- Support Jester dataset ([#864](https://github.com/open-mmlab/mmaction2/pull/864))

**Improvements**

- Add `metric_options` for evaluation to docs ([#873](https://github.com/open-mmlab/mmaction2/pull/873))
- Support creating a new label map based on custom classes for demos about spatio temporal demo ([#879](https://github.com/open-mmlab/mmaction2/pull/879))
- Improve document about AVA dataset preparation ([#878](https://github.com/open-mmlab/mmaction2/pull/878))
- Provide a script to extract clip-level feature ([#856](https://github.com/open-mmlab/mmaction2/pull/856))

**Bug and Typo Fixes**

- Fix issues about resume ([#877](https://github.com/open-mmlab/mmaction2/pull/877), [#878](https://github.com/open-mmlab/mmaction2/pull/878))
- Correct the key name of `eval_results` dictionary for metric 'mmit_mean_average_precision' ([#885](https://github.com/open-mmlab/mmaction2/pull/885))

**ModelZoo**

- Support Jester dataset ([#864](https://github.com/open-mmlab/mmaction2/pull/864))
- Support ACRN and Focal Loss ([#891](https://github.com/open-mmlab/mmaction2/pull/891))

### 0.14.0 (30/04/2021)

**Highlights**

- Support TRN
- Support Diving48

**New Features**

- Support TRN ([#755](https://github.com/open-mmlab/mmaction2/pull/755))
- Support Diving48 ([#835](https://github.com/open-mmlab/mmaction2/pull/835))
- Support Webcam Demo for Spatio-temporal Action Detection Models ([#795](https://github.com/open-mmlab/mmaction2/pull/795))

**Improvements**

- Add softmax option for pytorch2onnx tool ([#781](https://github.com/open-mmlab/mmaction2/pull/781))
- Support TRN ([#755](https://github.com/open-mmlab/mmaction2/pull/755))
- Test with onnx models and TensorRT engines ([#758](https://github.com/open-mmlab/mmaction2/pull/758))
- Speed up AVA Testing ([#784](https://github.com/open-mmlab/mmaction2/pull/784))
- Add `self.with_neck` attribute ([#796](https://github.com/open-mmlab/mmaction2/pull/796))
- Update installation document ([#798](https://github.com/open-mmlab/mmaction2/pull/798))
- Use a random master port ([#809](https://github.com/open-mmlab/mmaction2/pull/8098))
- Update AVA processing data document ([#801](https://github.com/open-mmlab/mmaction2/pull/801))
- Refactor spatio-temporal augmentation ([#782](https://github.com/open-mmlab/mmaction2/pull/782))
- Add QR code in CN README ([#812](https://github.com/open-mmlab/mmaction2/pull/812))
- Add Alternative way to download Kinetics ([#817](https://github.com/open-mmlab/mmaction2/pull/817), [#822](https://github.com/open-mmlab/mmaction2/pull/822))
- Refactor Sampler ([#790](https://github.com/open-mmlab/mmaction2/pull/790))
- Use EvalHook in MMCV with backward compatibility ([#793](https://github.com/open-mmlab/mmaction2/pull/793))
- Use MMCV Model Registry ([#843](https://github.com/open-mmlab/mmaction2/pull/843))

**Bug and Typo Fixes**

- Fix a bug in pytorch2onnx.py when `num_classes <= 4` ([#800](https://github.com/open-mmlab/mmaction2/pull/800), [#824](https://github.com/open-mmlab/mmaction2/pull/824))
- Fix `demo_spatiotemporal_det.py` error ([#803](https://github.com/open-mmlab/mmaction2/pull/803), [#805](https://github.com/open-mmlab/mmaction2/pull/805))
- Fix loading config bugs when resume ([#820](https://github.com/open-mmlab/mmaction2/pull/820))
- Make HMDB51 annotation generation more robust ([#811](https://github.com/open-mmlab/mmaction2/pull/811))

**ModelZoo**

- Update checkpoint for 256 height in something-V2 ([#789](https://github.com/open-mmlab/mmaction2/pull/789))
- Support Diving48 ([#835](https://github.com/open-mmlab/mmaction2/pull/835))

### 0.13.0 (31/03/2021)

**Highlights**

- Support LFB
- Support using backbone from MMCls/TorchVision
- Add Chinese documentation

**New Features**

- Support LFB ([#553](https://github.com/open-mmlab/mmaction2/pull/553))
- Support using backbones from MMCls for TSN ([#679](https://github.com/open-mmlab/mmaction2/pull/679))
- Support using backbones from TorchVision for TSN ([#720](https://github.com/open-mmlab/mmaction2/pull/720))
- Support Mixup and Cutmix for recognizers ([#681](https://github.com/open-mmlab/mmaction2/pull/681))
- Support Chinese documentation ([#665](https://github.com/open-mmlab/mmaction2/pull/665), [#680](https://github.com/open-mmlab/mmaction2/pull/680), [#689](https://github.com/open-mmlab/mmaction2/pull/689), [#701](https://github.com/open-mmlab/mmaction2/pull/701), [#702](https://github.com/open-mmlab/mmaction2/pull/702), [#703](https://github.com/open-mmlab/mmaction2/pull/703), [#706](https://github.com/open-mmlab/mmaction2/pull/706), [#716](https://github.com/open-mmlab/mmaction2/pull/716), [#717](https://github.com/open-mmlab/mmaction2/pull/717), [#731](https://github.com/open-mmlab/mmaction2/pull/731), [#733](https://github.com/open-mmlab/mmaction2/pull/733), [#735](https://github.com/open-mmlab/mmaction2/pull/735), [#736](https://github.com/open-mmlab/mmaction2/pull/736), [#737](https://github.com/open-mmlab/mmaction2/pull/737), [#738](https://github.com/open-mmlab/mmaction2/pull/738), [#739](https://github.com/open-mmlab/mmaction2/pull/739), [#740](https://github.com/open-mmlab/mmaction2/pull/740), [#742](https://github.com/open-mmlab/mmaction2/pull/742), [#752](https://github.com/open-mmlab/mmaction2/pull/752), [#759](https://github.com/open-mmlab/mmaction2/pull/759), [#761](https://github.com/open-mmlab/mmaction2/pull/761), [#772](https://github.com/open-mmlab/mmaction2/pull/772), [#775](https://github.com/open-mmlab/mmaction2/pull/775))

**Improvements**

- Add slowfast config/json/log/ckpt for training custom classes of AVA ([#678](https://github.com/open-mmlab/mmaction2/pull/678))
- Set RandAugment as Imgaug default transforms ([#585](https://github.com/open-mmlab/mmaction2/pull/585))
- Add `--test-last` & `--test-best` for `tools/train.py` to test checkpoints after training ([#608](https://github.com/open-mmlab/mmaction2/pull/608))
- Add fcn_testing in TPN ([#684](https://github.com/open-mmlab/mmaction2/pull/684))
- Remove redundant recall functions ([#741](https://github.com/open-mmlab/mmaction2/pull/741))
- Recursively remove pretrained step for testing ([#695](https://github.com/open-mmlab/mmaction2/pull/695))
- Improve demo by limiting inference fps ([#668](https://github.com/open-mmlab/mmaction2/pull/668))

**Bug and Typo Fixes**

- Fix a bug about multi-class in VideoDataset ([#723](https://github.com/open-mmlab/mmaction2/pull/678))
- Reverse key-value in anet filelist generation ([#686](https://github.com/open-mmlab/mmaction2/pull/686))
- Fix flow norm cfg typo ([#693](https://github.com/open-mmlab/mmaction2/pull/693))

**ModelZoo**

- Add LFB for AVA2.1 ([#553](https://github.com/open-mmlab/mmaction2/pull/553))
- Add TSN with ResNeXt-101-32x4d backbone as an example for using MMCls backbones ([#679](https://github.com/open-mmlab/mmaction2/pull/679))
- Add TSN with Densenet161 backbone as an example for using TorchVision backbones ([#720](https://github.com/open-mmlab/mmaction2/pull/720))
- Add slowonly_nl_embedded_gaussian_r50_4x16x1_150e_kinetics400_rgb ([#690](https://github.com/open-mmlab/mmaction2/pull/690))
- Add slowonly_nl_embedded_gaussian_r50_8x8x1_150e_kinetics400_rgb ([#704](https://github.com/open-mmlab/mmaction2/pull/704))
- Add slowonly_nl_kinetics_pretrained_r50_4x16x1(8x8x1)_20e_ava_rgb ([#730](https://github.com/open-mmlab/mmaction2/pull/730))

### 0.12.0 (28/02/2021)

**Highlights**

- Support TSM-MobileNetV2
- Support TANet
- Support GPU Normalize

**New Features**

- Support TSM-MobileNetV2 ([#415](https://github.com/open-mmlab/mmaction2/pull/415))
- Support flip with label mapping ([#591](https://github.com/open-mmlab/mmaction2/pull/591))
- Add seed option for sampler ([#642](https://github.com/open-mmlab/mmaction2/pull/642))
- Support GPU Normalize ([#586](https://github.com/open-mmlab/mmaction2/pull/586))
- Support TANet ([#595](https://github.com/open-mmlab/mmaction2/pull/595))

**Improvements**

- Training custom classes of ava dataset ([#555](https://github.com/open-mmlab/mmaction2/pull/555))
- Add CN README in homepage ([#592](https://github.com/open-mmlab/mmaction2/pull/592), [#594](https://github.com/open-mmlab/mmaction2/pull/594))
- Support soft label for CrossEntropyLoss ([#625](https://github.com/open-mmlab/mmaction2/pull/625))
- Refactor config: Specify `train_cfg` and `test_cfg` in `model` ([#629](https://github.com/open-mmlab/mmaction2/pull/629))
- Provide an alternative way to download older kinetics annotations ([#597](https://github.com/open-mmlab/mmaction2/pull/597))
- Update FAQ for
  - 1). data pipeline about video and frames ([#598](https://github.com/open-mmlab/mmaction2/pull/598))
  - 2). how to show results ([#598](https://github.com/open-mmlab/mmaction2/pull/598))
  - 3). batch size setting for batchnorm ([#657](https://github.com/open-mmlab/mmaction2/pull/657))
  - 4). how to fix stages of backbone when finetuning models ([#658](https://github.com/open-mmlab/mmaction2/pull/658))
- Modify default value of `save_best` ([#600](https://github.com/open-mmlab/mmaction2/pull/600))
- Use BibTex rather than latex in markdown ([#607](https://github.com/open-mmlab/mmaction2/pull/607))
- Add warnings of uninstalling mmdet and supplementary documents ([#624](https://github.com/open-mmlab/mmaction2/pull/624))
- Support soft label for CrossEntropyLoss ([#625](https://github.com/open-mmlab/mmaction2/pull/625))

**Bug and Typo Fixes**

- Fix value of `pem_low_temporal_iou_threshold` in BSN ([#556](https://github.com/open-mmlab/mmaction2/pull/556))
- Fix ActivityNet download script ([#601](https://github.com/open-mmlab/mmaction2/pull/601))

**ModelZoo**

- Add TSM-MobileNetV2 for Kinetics400 ([#415](https://github.com/open-mmlab/mmaction2/pull/415))
- Add deeper SlowFast models ([#605](https://github.com/open-mmlab/mmaction2/pull/605))

### 0.11.0 (31/01/2021)

**Highlights**

- Support imgaug
- Support spatial temporal demo
- Refactor EvalHook, config structure, unittest structure

**New Features**

- Support [imgaug](https://imgaug.readthedocs.io/en/latest/index.html) for augmentations in the data pipeline ([#492](https://github.com/open-mmlab/mmaction2/pull/492))
- Support setting `max_testing_views` for extremely large models to save GPU memory used ([#511](https://github.com/open-mmlab/mmaction2/pull/511))
- Add spatial temporal demo ([#547](https://github.com/open-mmlab/mmaction2/pull/547), [#566](https://github.com/open-mmlab/mmaction2/pull/566))

**Improvements**

- Refactor EvalHook ([#395](https://github.com/open-mmlab/mmaction2/pull/395))
- Refactor AVA hook ([#567](https://github.com/open-mmlab/mmaction2/pull/567))
- Add repo citation ([#545](https://github.com/open-mmlab/mmaction2/pull/545))
- Add dataset size of Kinetics400 ([#503](https://github.com/open-mmlab/mmaction2/pull/503))
- Add lazy operation docs ([#504](https://github.com/open-mmlab/mmaction2/pull/504))
- Add class_weight for CrossEntropyLoss and BCELossWithLogits ([#509](https://github.com/open-mmlab/mmaction2/pull/509))
- add some explanation about the resampling in slowfast ([#502](https://github.com/open-mmlab/mmaction2/pull/502))
- Modify paper title in README.md ([#512](https://github.com/open-mmlab/mmaction2/pull/512))
- Add alternative ways to download Kinetics ([#521](https://github.com/open-mmlab/mmaction2/pull/521))
- Add OpenMMLab projects link in README ([#530](https://github.com/open-mmlab/mmaction2/pull/530))
- Change default preprocessing to shortedge to 256 ([#538](https://github.com/open-mmlab/mmaction2/pull/538))
- Add config tag in dataset README ([#540](https://github.com/open-mmlab/mmaction2/pull/540))
- Add solution for markdownlint installation issue ([#497](https://github.com/open-mmlab/mmaction2/pull/497))
- Add dataset overview in readthedocs ([#548](https://github.com/open-mmlab/mmaction2/pull/548))
- Modify the trigger mode of the warnings of missing mmdet ([#583](https://github.com/open-mmlab/mmaction2/pull/583))
- Refactor config structure ([#488](https://github.com/open-mmlab/mmaction2/pull/488), [#572](https://github.com/open-mmlab/mmaction2/pull/572))
- Refactor unittest structure ([#433](https://github.com/open-mmlab/mmaction2/pull/433))

**Bug and Typo Fixes**

- Fix a bug about ava dataset validation ([#527](https://github.com/open-mmlab/mmaction2/pull/527))
- Fix a bug about ResNet pretrain weight initialization ([#582](https://github.com/open-mmlab/mmaction2/pull/582))
- Fix a bug in CI due to MMCV index ([#495](https://github.com/open-mmlab/mmaction2/pull/495))
- Remove invalid links of MiT and MMiT ([#516](https://github.com/open-mmlab/mmaction2/pull/516))
- Fix frame rate bug for AVA preparation ([#576](https://github.com/open-mmlab/mmaction2/pull/576))

**ModelZoo**

### 0.10.0 (31/12/2020)

**Highlights**

- Support Spatio-Temporal Action Detection (AVA)
- Support precise BN

**New Features**

- Support precise BN ([#501](https://github.com/open-mmlab/mmaction2/pull/501/))
- Support Spatio-Temporal Action Detection (AVA) ([#351](https://github.com/open-mmlab/mmaction2/pull/351))
- Support to return feature maps in `inference_recognizer` ([#458](https://github.com/open-mmlab/mmaction2/pull/458))

**Improvements**

- Add arg `stride` to long_video_demo.py, to make inference faster ([#468](https://github.com/open-mmlab/mmaction2/pull/468))
- Support training and testing for Spatio-Temporal Action Detection ([#351](https://github.com/open-mmlab/mmaction2/pull/351))
- Fix CI due to pip upgrade ([#454](https://github.com/open-mmlab/mmaction2/pull/454))
- Add markdown lint in pre-commit hook ([#255](https://github.com/open-mmlab/mmaction2/pull/225))
- Speed up confusion matrix calculation ([#465](https://github.com/open-mmlab/mmaction2/pull/465))
- Use title case in modelzoo statistics ([#456](https://github.com/open-mmlab/mmaction2/pull/456))
- Add FAQ documents for easy troubleshooting. ([#413](https://github.com/open-mmlab/mmaction2/pull/413), [#420](https://github.com/open-mmlab/mmaction2/pull/420), [#439](https://github.com/open-mmlab/mmaction2/pull/439))
- Support Spatio-Temporal Action Detection with context ([#471](https://github.com/open-mmlab/mmaction2/pull/471))
- Add class weight for CrossEntropyLoss and BCELossWithLogits ([#509](https://github.com/open-mmlab/mmaction2/pull/509))
- Add Lazy OPs docs ([#504](https://github.com/open-mmlab/mmaction2/pull/504))

**Bug and Typo Fixes**

- Fix typo in default argument of BaseHead ([#446](https://github.com/open-mmlab/mmaction2/pull/446))
- Fix potential bug about `output_config` overwrite ([#463](https://github.com/open-mmlab/mmaction2/pull/463))

**ModelZoo**

- Add SlowOnly, SlowFast for AVA2.1 ([#351](https://github.com/open-mmlab/mmaction2/pull/351))

### 0.9.0 (30/11/2020)

**Highlights**

- Support GradCAM utils for recognizers
- Support ResNet Audio model

**New Features**

- Automatically add modelzoo statistics to readthedocs ([#327](https://github.com/open-mmlab/mmaction2/pull/327))
- Support GYM99 ([#331](https://github.com/open-mmlab/mmaction2/pull/331), [#336](https://github.com/open-mmlab/mmaction2/pull/336))
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
