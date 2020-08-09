
# MMAction2 Release Notes

## Master

### Models
* Add NonLocal module in TSM and I3D [#41](https://github.com/open-mmlab/mmaction2/pull/41)
* Support SSN
    - Add pipeline [#33](https://github.com/open-mmlab/mmaction2/pull/33)
    - Add losses [#52](https://github.com/open-mmlab/mmaction2/pull/52)
* Add CSN model [#87](https://github.com/open-mmlab/mmaction2/pull/87)

### Datasets

* Support hmdb51 dataset preparation [#60](https://github.com/open-mmlab/mmaction2/pull/60)

### New Features

* Support rawframe inference in demo [#59](https://github.com/open-mmlab/mmaction2/pull/59)
* Support output video or gif in demo [#72](https://github.com/open-mmlab/mmaction2/pull/72)
* Add script to encode videos [#84](https://github.com/open-mmlab/mmaction2/pull/84)
* Support `with_offset` for rawframe dataset [#48](https://github.com/open-mmlab/mmaction2/pull/48)

### Modelzoo

* Update Slowfast [#51](https://github.com/open-mmlab/mmaction2/pull/51)
* Update TSN, TSM video checkpoints [#50](https://github.com/open-mmlab/mmaction2/pull/50)
* Add data benchmark for TSN [#57](https://github.com/open-mmlab/mmaction2/pull/57)
* Add slowonly data benchmark [#77](https://github.com/open-mmlab/mmaction2/pull/77)

### Bug Fixes & Improvements

* `CosineAnnealing` Typo [#47](https://github.com/open-mmlab/mmaction2/pull/47)
* Bug fix in analyze_log [#54](https://github.com/open-mmlab/mmaction2/pull/54)
* Fix configs for localization test [#67](https://github.com/open-mmlab/mmaction2/pull/67)
* Fix the bug of generating HMDB51 class index file [#69](https://github.com/open-mmlab/mmaction2/pull/69)
* Fix demo [#88](https://github.com/open-mmlab/mmaction2/pull/88)
* Correct the sthv1/sthv2 rawframes filelist generate command [#71](https://github.com/open-mmlab/mmaction2/pull/71)
* Polish data preparation codes [#70](https://github.com/open-mmlab/mmaction2/pull/70)
* Add `target_resolution` in demo and fix some args [#80](https://github.com/open-mmlab/mmaction2/pull/80)
* Improve data preparation code [#58](https://github.com/open-mmlab/mmaction2/pull/58)
* Improve unittest coverage and minor fix [#62](https://github.com/open-mmlab/mmaction2/pull/62)

### Docs

* Fix bsn configs filename in README.md [#74](https://github.com/open-mmlab/mmaction2/pull/74)
* Update docs for demo [#81](https://github.com/open-mmlab/mmaction2/pull/81)

### Notable Changes

* Use `xxInit()` method to get total_frames [#90](https://github.com/open-mmlab/mmaction2/pull/90)
* Polish benchmark [#43](https://github.com/open-mmlab/mmaction2/pull/43)
* Use `RawFrameDecode` to replace `FrameSelector` [#91](https://github.com/open-mmlab/mmaction2/pull/91)
* Move `start_index` from `SampleFrames` to dataset level [#89](https://github.com/open-mmlab/mmaction2/pull/89)

## v0.5.0 Initial Release (2020-07-21)
