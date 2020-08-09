
# MMAction2 Release Notes

## v0.6.0 (Master)

### Models
* Add NonLocal module in TSM and I3D (https://github.com/open-mmlab/mmaction2/pull/41)
* Add SSN pipeline (https://github.com/open-mmlab/mmaction2/pull/33)
* Add SSN losses (https://github.com/open-mmlab/mmaction2/pull/52)
* Add CSN model (https://github.com/open-mmlab/mmaction2/pull/87)

### Datasets

* Support hmdb51 dataset preparation (https://github.com/open-mmlab/mmaction2/pull/60)

### New Features

* Support rawframe inference in demo (https://github.com/open-mmlab/mmaction2/pull/59)
* Support output video or gif in demo (https://github.com/open-mmlab/mmaction2/pull/72)
* Add script to encode videos (https://github.com/open-mmlab/mmaction2/pull/84)
* Support `with_offset` for rawframe dataset (https://github.com/open-mmlab/mmaction2/pull/48)

### Modelzoo

* Update Slowfast (https://github.com/open-mmlab/mmaction2/pull/51)
* Update TSN, TSM video checkpoints (https://github.com/open-mmlab/mmaction2/pull/50)
* Add data benchmark for TSN (https://github.com/open-mmlab/mmaction2/pull/57)
* Add slowonly data benchmark (https://github.com/open-mmlab/mmaction2/pull/77)

### Bug Fixes & Improvements

* `CosineAnnealing` Typo (https://github.com/open-mmlab/mmaction2/pull/47)
* Bug fix in analyze_log (https://github.com/open-mmlab/mmaction2/pull/54)
* Fix configs for localization test (https://github.com/open-mmlab/mmaction2/pull/67)
* Fix the bug of generating HMDB51 class index file (https://github.com/open-mmlab/mmaction2/pull/69)
* Fix demo (https://github.com/open-mmlab/mmaction2/pull/88)
* Correct the sthv1/sthv2 rawframes filelist generate command (https://github.com/open-mmlab/mmaction2/pull/71)
* Polish data preparation codes (https://github.com/open-mmlab/mmaction2/pull/70)
* Add `target_resolution` in demo and fix some args (https://github.com/open-mmlab/mmaction2/pull/80)
* Improve data preparation code (https://github.com/open-mmlab/mmaction2/pull/58)
* Improve unittest coverage and minor fix (https://github.com/open-mmlab/mmaction2/pull/62)

### Docs

* Fix bsn configs filename in README.md (https://github.com/open-mmlab/mmaction2/pull/74)
* Update docs for demo (https://github.com/open-mmlab/mmaction2/pull/81)

### Misc

* Use `xxInit()` method to get total_frames (https://github.com/open-mmlab/mmaction2/pull/90)
* Polish benchmark (https://github.com/open-mmlab/mmaction2/pull/43)
* Use `RawFrameDecode` to replace `FrameSelector` (https://github.com/open-mmlab/mmaction2/pull/91)
* Move `start_index` from `SampleFrames` to dataset level (https://github.com/open-mmlab/mmaction2/pull/89)

## v0.5.0 Initial Release (2020-07-21)
