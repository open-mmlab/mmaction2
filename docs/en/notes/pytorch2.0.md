# PyTorch 2.0 Compatibility and Benchmark

PyTorch introduced `torch.compile` in its 2.0 release. It compiles your model to speedup trainning & validation. We provide a benchmark result and compatibility of typical models in MMAction2. Except for one model (MViT) that fails to compile, the performance of other models remains consistent before and after compilation.

| Config                                                                    | compiled | Train time / iter (s) | GPU memory (M) | test metric  |
| ------------------------------------------------------------------------- | -------- | --------------------- | -------------- | ------------ |
| tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb                    | False    | 0.50                  | 42537          | 36.55        |
| tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb                    | True     | 0.61                  | 53149          | 36.72        |
| timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb                         | False    | 0.688                 | 14263          | 77.69        |
| timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb                         | True     | 0.691                 | 13863          | 77.57        |
| stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d                          | False    | 0.0305                | 1184           | 91.69        |
| stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d                          | True     | 0.0298                | 1273           | 91.64        |
| slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint                           | False    | 0.498                 | 9581           | 93.6         |
| slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint                           | True     | 0.505                 | 11968          | 93.49        |
| slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb            | False    | 0.17                  | 8278           | 20.76        |
| slowonly_kinetics400-pretrained-r50_8xb16-4x16x1-20e_ava21-rgb            | True     | 0.1835                | 12004          | 21.67        |
| swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb          | False    | 0.323                 | 21651          | 78.90        |
| swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb          | True     | 0.262                 | 20905          | 78.70        |
| slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb | False    | 0.098                 | 5777           | 75.12        |
| slowonly_imagenet-pretrained-r50_8xb16-4x16x1-steplr-150e_kinetics400-rgb | True     | 0.0942                | 7095           | 75.15        |
| mvit-small-p244_32xb16-16x4x1-200e_kinetics400-rgb                        | Fail     | incompatible          | incompatible   | incompatible |
