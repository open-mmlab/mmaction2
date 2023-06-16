# 准备 HACS Segments

## 简介

<!-- [DATASET] -->

```BibTeX
@inproceedings{zhao2019hacs,
  title={Hacs: Human action clips and segments dataset for recognition and temporal localization},
  author={Zhao, Hang and Torralba, Antonio and Torresani, Lorenzo and Yan, Zhicheng},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8668--8678},
  year={2019}
}
```

### 0. 下载视频

在我们开始准备数据集之前，请按照[官方代码库](https://github.com/hangzhaomit/HACS-dataset)的指令下载HACS Segments数据集中的视频。如果有视频缺失，您可以向HACS数据集存储库的维护者提交请求以获取缺失的视频。但是如果一些视频缺失，您仍然可以为MMAction2准备数据集。

在下载完数据集后，请将数据集文件夹移动到(或者使用软链接)`$MMACTION2/tools/data/hacs/`。文件夹结构应该如下所示：

```
mmaction2
├── mmaction
├── data
├── configs
├── tools
│   ├── hacs
│   │   ├── slowonly_feature_infer.py
│   │   ├── ..
│   │   ├── data
│   │   │   ├── Applying_sunscreen
│   │   │   │   ├── v_0Ch__DqMPwA.mp4
│   │   │   │   ├── v_9CTDjFHl8WE.mp4
│   │   │   │   ├── ..


```

在开始之前，请确保您位于`$MMACTION2/tools/data/hacs/`路径下。

### 1. 提取特征

以下是使用[SlowOnly ResNet50 8x8](/configs/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-4x16x1-steplr-150e_kinetics700-rgb.py)在Kinetics700数据集上预训练的模型，从HACS视频中提取特征。对于每个视频，我们均匀采样100个视频片段，并提取700维输出（softmax之前）作为特征，即特征形状为100x700。

首先，我们使用如下命令生成数据集的视频列表：

```
python generate_list.py
```

这将生成一个位于`$MMACTION2/tools/data/hacs/`的`hacs_data.txt`文件，其内容格式如下：

```
Horseback_riding/v_Sr2BSq_8FMw.mp4 0
Horseback_riding/v_EQb6OKoqz3Q.mp4 1
Horseback_riding/v_vYKUV8TRngg.mp4 2
Horseback_riding/v_Y8U0X1F-0ck.mp4 3
Horseback_riding/v_hnspbB7wNh0.mp4 4
Horseback_riding/v_HPhlhrT9IOk.mp4 5
```

接下来，我们使用[slowonly_feature_infer.py](/tools/data/hacs/slowonly_feature_infer.py) 配置文件来提取特征：

```
# 指定提取特征的GPU数量
NUM_GPUS=8

# 下载预训练模型权重
wget https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb/slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb_20221013-15b93b10.pth

bash ../mmaction2/tools/dist_test.sh \
    slowonly_feature_infer.py \
    slowonly_imagenet-pretrained-r50_16xb16-8x8x1-steplr-150e_kinetics700-rgb_20221013-15b93b10.pth \
    $NUM_GPUS --dump result.pkl
```

我们将得到一个名为 `result.pkl` 的文件，其中包含每个视频的大小为100x700的特征。我们将特征重写为csv格式，并保存在 `$MMACTION2/data/HACS/` 目录下。

```
＃确保您位于$ $MMACTION2/tools/data/hacs/
python write_feature_csv.py
```

### 2. 准备标注文件

我们首先从官方仓库下载标注文件：

```
wget https://github.com/hangzhaomit/HACS-dataset/raw/master/HACS_v1.1.1.zip
unzip HACS_v1.1.1.zip
```

解压缩后，应该有一个名为`HACS_v1.1.1`的文件夹，其中包含一个名为`HACS_segments_v1.1.1.json`的文件。

我们在`$MMACTION2/data/HACS/`目录下生成`hacs_anno_train.json`、`hacs_anno_val.json`和`hacs_anno_test.json`文件：

```
python3 generate_anotations.py
```

完成这两个步骤后，HACS Segments数据集的文件夹结构应该如下所示：

```
mmaction2
├── mmaction
├── data
│   ├── HACS
│   │   ├── hacs_anno_train.json
│   │   ├── hacs_anno_val.json
│   │   ├── hacs_anno_test.json
│   │   ├── slowonly_feature
│   │   │   ├── v_008gY2B8Pf4.csv
│   │   │   ├── v_0095rqic1n8.csv
├── configs
├── tools

```
