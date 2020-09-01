## Changelog

### v0.6.0 (20/8/2020)

**Highlights**
- Support NonLocal module in TSM and I3D ([#41](https://github.com/open-mmlab/mmaction2/pull/41))
- Support SSN ([#33](https://github.com/open-mmlab/mmaction2/pull/33), [#52](https://github.com/open-mmlab/mmaction2/pull/52))
- Support CSN model ([#87](https://github.com/open-mmlab/mmaction2/pull/87))
- Support hmdb51 dataset preparation ([#60](https://github.com/open-mmlab/mmaction2/pull/60))
- Enhance demo by supporting rawframe inference ([#59](https://github.com/open-mmlab/mmaction2/pull/59)),
  output video/gif ([#72](https://github.com/open-mmlab/mmaction2/pull/72)),
- Support encode videos from frames ([#84](https://github.com/open-mmlab/mmaction2/pull/84))

**Bug Fixes**
- `CosineAnnealing` Typo ([#47](https://github.com/open-mmlab/mmaction2/pull/47))
- Bug fix in analyze_log ([#54](https://github.com/open-mmlab/mmaction2/pull/54))
- Fix configs for localization test ([#67](https://github.com/open-mmlab/mmaction2/pull/67))
- Fix the bug of generating HMDB51 class index file ([#69](https://github.com/open-mmlab/mmaction2/pull/69))
- Fix the bug of using `loac_checkpoint()` in ResNet ([#93](https://github.com/open-mmlab/mmaction2/pull/93))
- Fix the bug of `--work-dir` when using slurm training script ([#110](https://github.com/open-mmlab/mmaction2/pull/110))
- Correct the sthv1/sthv2 rawframes filelist generate command ([#71](https://github.com/open-mmlab/mmaction2/pull/71))

**New Features**
- Support `with_offset` for rawframe dataset ([#48](https://github.com/open-mmlab/mmaction2/pull/48))
- Update Slowfast modelzoo ([#51](https://github.com/open-mmlab/mmaction2/pull/51))
- Update TSN, TSM video checkpoints ([#50](https://github.com/open-mmlab/mmaction2/pull/50))
- Add data benchmark for TSN ([#57](https://github.com/open-mmlab/mmaction2/pull/57))
- Add slowonly data benchmark ([#77](https://github.com/open-mmlab/mmaction2/pull/77))
- Add BSN/BMN performance results with feature extracted by our codebase ([#99](https://github.com/open-mmlab/mmaction2/pull/99))

**Improvements**
- Polish data preparation codes ([#70](https://github.com/open-mmlab/mmaction2/pull/70))
- Improve data preparation scripts ([#58](https://github.com/open-mmlab/mmaction2/pull/58))
- Improve unittest coverage and minor fix ([#62](https://github.com/open-mmlab/mmaction2/pull/62))
- Support `multi-class` in TSMHead ([#104](https://github.com/open-mmlab/mmaction2/pull/104))
- Use `xxInit()` method to get `total_frames` and make `total_frames` a required key ([#90](https://github.com/open-mmlab/mmaction2/pull/90))

### v0.5.0 (9/7/2020)

**Highlights**
- MMAction2 is released

**New Features**
- Support various datasets: UCF101, Kinetics-400, Something-Something V1&V2, Moments in Time,
  Multi-Moments in Time, THUMOS14
- Support various action recognition methods: TSN, TSM, R(2+1)D, I3D, SlowOnly, SlowFast, Non-local
- Support various action localization methods: BSN, BMN
- Colab demo for action recognition
