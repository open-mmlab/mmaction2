# RGBPoseConv3D

## Introduction

RGBPoseConv3D is a framework that jointly use 2D human skeletons and RGB appearance for human action recognition. It is a 3D CNN with two streams, with the architecture borrowed from SlowFast. In RGBPoseConv3D:

- The RGB stream corresponds to the `slow` stream in SlowFast; The Skeleton stream corresponds to the `fast` stream in SlowFast.
- The input resolution of RGB frames is `4x` larger than the pseudo heatmaps.
- Bilateral connections are used for early feature fusion between the two modalities.

<div align=center>
<img src="https://user-images.githubusercontent.com/34324155/209961351-6def0074-9b05-43fc-8210-a1cdaaed6536.png" width=70%/>
</div>

## Citation

```BibTeX
@inproceedings{duan2022revisiting,
  title={Revisiting skeleton-based action recognition},
  author={Duan, Haodong and Zhao, Yue and Chen, Kai and Lin, Dahua and Dai, Bo},
  booktitle={CVPR},
  pages={2969--2978},
  year={2022}
}
```

## How to train RGBPoseConv3D (on NTURGB+D, for example)?

#### Step 0. Data Preparation

Besides the skeleton annotations, you also need RGB videos to train RGBPoseConv3D. You need to download them from the official website of [NTURGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and put these videos in `$MMACTION2/data/nturgbd_raw`. After that, you need to use the provided script to compress the raw videos (from `1920x1080` to `960x540`) and switch the suffix to `.mp4`:

```bash
# That step is mandatory, unless you know how to modify the code & config to make it work for raw videos!
python tools/data/skeleton/compress_nturgbd.py
```

After that, you will find processed videos in `$MMACTION2/data/nturgbd_videos`, named like `S001C001P001R001A001.mp4`.

#### Step 1. Pretraining

You first need to train the RGB-only and Pose-only model on the target dataset, the pretrained checkpoints will be used to initialize the RGBPoseConv3D model.

You can either train these two models from scratch with provided configs files:

```bash
# We train each model for 180 epochs. By default, we use 8 GPUs.
# Train the RGB-only model
bash tools/dist_train.sh configs/skeleton/posec3d/rgbpose_conv3d/rgb_only.py 8
# Train the Pose-only model
bash tools/dist_train.sh configs/skeleton/posec3d/rgbpose_conv3d/pose_only.py 8
```

or directly download and use the provided pretrain models:

|    Dataset    |                                Config                                |                                 Checkpoint                                 | Top-1 (1 clip testing) | Top-1 (10 clip testing) |
| :-----------: | :------------------------------------------------------------------: | :------------------------------------------------------------------------: | :--------------------: | :---------------------: |
| NTURGB+D XSub |  [rgb_config](/configs/skeleton/posec3d/rgbpose_conv3d/rgb_only.py)  | [rgb_ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgb_only_20230228-576b9f86.pth) |          94.9          |          95.4           |
| NTURGB+D XSub | [pose_config](/configs/skeleton/posec3d/rgbpose_conv3d/pose_only.py) | [pose_ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/pose_only_20230228-fa40054e.pth) |          93.1          |          93.5           |

#### Step 2. Generate the initializing weight for RGBPoseConv3D

You can use the provided [IPython notebook](/configs/skeleton/posec3d/rgbpose_conv3d/merge_pretrain.ipynb) to merge two pretrained models into a single `rgbpose_conv3d_init.pth`.

You can do it your own or directly download and use the provided [rgbpose_conv3d_init.pth](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_init_20230228-09b7684b.pth).

#### Step 3. Finetune RGBPoseConv3D

You can use our provided config files to finetune RGBPoseConv3D, jointly with two modalities (RGB & Pose):

```bash
# We finetune RGBPoseConv3D for 20 epochs on NTURGB+D XSub (8 GPUs)
bash tools/dist_train.sh configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py 8
# After finetuning, you can test the model with the following command (8 GPUs)
bash tools/dist_test.sh configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py $CKPT 8 --dump result.pkl
```

**Notes**

1. We use linear scaling learning rate (`Initial LR` ∝ `Batch Size`). If you change the training batch size, remember to change the initial LR proportionally.

2. Though optimized, multi-clip testing may consumes large amounts of time. For faster inference, you may change the test_pipeline to disable the multi-clip testing, this may lead to a small drop in recognition performance. Below is the guide:

   ```python
   test_pipeline = [
       dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=10, test_mode=True), # change `num_clips=10` to `num_clips=1`
       dict(type='MMDecode'),
       dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
       dict(type='Resize', scale=(256, 256), keep_ratio=False),
       dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
       dict(type='FormatShape', input_format='NCTHW'),
       dict(type='PackActionInputs', collect_keys=('imgs', 'heatmap_imgs'))
   ]
   ```

## Results

On action recognition with multiple modalities (RGB & Pose), RGBPoseConv3D can achieve better recognition performance than the late fusion baseline.

|    Dataset    |        Fusion         |        Config         |         Checkpoint         | RGB Stream Top-1<br>(1-clip / 10-clip) | Pose Stream Top-1<br>(1-clip / 10-clip) | 2 Stream Top-1 (1:1)<br>(1-clip / 10-clip) |
| :-----------: | :-------------------: | :-------------------: | :------------------------: | :------------------------------------: | :-------------------------------------: | :----------------------------------------: |
| NTURGB+D XSub |      Late Fusion      | [rgb_config](/configs/skeleton/posec3d/rgbpose_conv3d/rgb_only.py)<br>[pose_config](/configs/skeleton/posec3d/rgbpose_conv3d/pose_only.py) | [rgb_ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgb_only_20230228-576b9f86.pth)<br>[pose_ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/pose_only_20230228-fa40054e.pth) |              94.9 / 95.4               |               93.1 / 93.5               |                96.0 / 96.2                 |
| NTURGB+D XSub | Early Fusion + Late Fusion | [config](/configs/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d.py) | [ckpt](https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/rgbpose_conv3d/rgbpose_conv3d_20230301-ac7b0e77.pth) |              96.2 / 96.4               |               96.0 / 96.2               |                96.6 / 96.8                 |

**Notes**

For both `Late Fusion` and `Early Fusion + Late Fusion`, we combine the action scores based on two modalities with 1:1 ratio to get the final prediction.
