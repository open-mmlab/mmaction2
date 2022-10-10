# C2D

<!-- [ALGORITHM] -->

C2D 是 [Non-local Neural Networks](https://arxiv.org/abs/1711.07971) 的基准模型

注意: C2D的实现在 1.上述文章; 2."SlowFast"仓库; 3."Video-Nonlocal-Net"仓库； 三者稍有不同

MMAction2 中的 C2D 与 ["Video-Nonlocal-Net"仓库](https://github.com/facebookresearch/video-nonlocal-net/tree/main/scripts/run_c2d_baseline_400k.sh)保持一致

具体地:

- maxpool3d_1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
- maxpool3d_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

MMAction2 中的 C2D_Nopool 与 ["SlowFast"仓库](https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml)保持一致
