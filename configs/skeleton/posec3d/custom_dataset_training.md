# Custom Dataset Training with PoseC3D

We provide a step-by-step tutorial on how to train your custom dataset with PoseC3D.

1. First, you should know that action recognition with PoseC3D requires skeleton information only and for that you need to prepare your custom annotation files (for training and validation). To start with, you need to install MMDetection and MMPose. Then you need to take advantage of [ntu_pose_extraction.py](https://github.com/open-mmlab/mmaction2/blob/90fc8440961987b7fe3ee99109e2c633c4e30158/tools/data/skeleton/ntu_pose_extraction.py) as shown in [Prepare Annotations](https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md#prepare-annotations) to extract 2D keypoints for each video in your custom dataset. The command looks like (assuming the name of your video is `some_video_from_my_dataset.mp4`):

   ```shell
   # You can use the above command to generate pickle files for all of your training and validation videos.
   python ntu_pose_extraction.py some_video_from_my_dataset.mp4 some_video_from_my_dataset.pkl
   ```

   @kennymckormick's [note](https://github.com/open-mmlab/mmaction2/issues/1216#issuecomment-950130079):

   > One only thing you may need to change is that: since ntu_pose_extraction.py is developed specifically for pose extraction of NTU videos, you can skip the [ntu_det_postproc](https://github.com/open-mmlab/mmaction2/blob/90fc8440961987b7fe3ee99109e2c633c4e30158/tools/data/skeleton/ntu_pose_extraction.py#L307) step when using this script for extracting pose from your custom video datasets.

2. Then, you will collect all the pickle files into one list for training (and, of course, for validation) and save them as a single file (like `custom_dataset_train.pkl` or `custom_dataset_val.pkl`). At that time, you finalize preparing annotation files for your custom dataset.

3. Next, you may use the following script (with some alterations according to your needs) for training as shown in [PoseC3D/Train](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/README.md#train): `python tools/train.py configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py --work-dir work_dirs/slowonly_r50_u48_240e_ntu120_xsub_keypoint --validate --test-best --gpus 2 --seed 0 --deterministic`:

   - Before running the above script, you need to modify the variables to initialize with your newly made annotation files:

     ```python
     model = dict(
         ...
         cls_head=dict(
             ...
             num_classes=4,    # Your class number
             ...
         ),
         ...
     )

     ann_file_train = 'data/posec3d/custom_dataset_train.pkl'  # Your annotation for training
     ann_file_val = 'data/posec3d/custom_dataset_val.pkl'      # Your annotation for validation

     load_from = 'pretrained_weight.pth'       # Your can use released weights for initialization, set to None if training from scratch

     # You can also alter the hyper parameters or training schedule
     ```

With that, your machine should start its work to let you grab a cup of coffee and watch how the training goes.
