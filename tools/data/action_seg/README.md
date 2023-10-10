# Preparing Datasets for Action Segmentation

## Introduction

<!-- [DATASET] -->

```BibTeX
@inproceedings{fathi2011learning,
  title={Learning to recognize objects in egocentric activities},
  author={Fathi, Alireza and Ren, Xiaofeng and Rehg, James M},
  booktitle={CVPR 2011},
  pages={3281--3288},
  year={2011},
  organization={IEEE}
}
```

```BibTeX
@inproceedings{stein2013combining,
  title={Combining embedded accelerometers with computer vision for recognizing food preparation activities},
  author={Stein, Sebastian and McKenna, Stephen J},
  booktitle={Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing},
  pages={729--738},
  year={2013}
}
```

```BibTeX
@inproceedings{kuehne2014language,
  title={The language of actions: Recovering the syntax and semantics of goal-directed human activities},
  author={Kuehne, Hilde and Arslan, Ali and Serre, Thomas},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={780--787},
  year={2014}
}
```

For basic dataset information, you can refer to the articles.
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/`.
To run the bash scripts below, you need to install `unzip`. you can install it by `sudo apt-get install unzip`.

## Step 1. Prepare Annotations and Features

First of all, you can run the following script to prepare annotations and features.

```shell
bash download_datasets.sh
```

## Step 2. Preprocess the Data

you can execute the following script to preprocess the downloaded data and generate two folders for each dataset, 'gt_arr' and 'gt_boundary_arr'.

```shell
python tools/data/action_seg/generate_boundary_array.py --dataset-dir action_seg
python tools/data/action_seg/generate_gt_array.py --dataset_dir data/action_seg
```

## Step 3. Check Directory Structure

After the whole data process for GTEA, 50Salads and Breakfast preparation,
you will get the features, splits ,annotation files and groundtruth boundaries for the datasets.

For extracting features from your own videos, please refer to [activitynet](/tools/data/activitynet/README.md).

In the context of the whole project (for GTEA, 50Salads and Breakfast), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── action_seg
│   │   ├── gtea
│   │   │   ├── features
│   │   │   │   ├── S1_Cheese_C1.npy
│   │   │   │   ├── S1_Coffee_C1.npy
│   │   │   │   ├── ...
│   │   │   ├── groundTruth
│   │   │   │   ├── S1_Cheese_C1.txt
│   │   │   │   ├── S1_Coffee_C1.txt
│   │   │   │   ├── ...
│   │   │   ├── gt_arr
│   │   │   │   ├── S1_Cheese_C1.npy
│   │   │   │   ├── S1_Coffee_C1.npy
│   │   │   │   ├── ...
│   │   │   ├── gt_boundary_arr
│   │   │   │   ├── S1_Cheese_C1.npy
│   │   │   │   ├── S1_Coffee_C1.npy
│   │   │   │   ├── ...
│   │   │   ├── splits
│   │   │   │   ├── fifa_mean_dur_split1.pt
│   │   │   │   ├── fifa_mean_dur_split2.pt
│   │   │   │   ├── ...
│   │   │   │   ├── test.split0.bundle
│   │   │   │   ├── test.split1.bundle
│   │   │   │   ├── ...
│   │   │   │   ├── train.split0.bundle
│   │   │   │   ├── train.split1.bundle
│   │   │   │   ├── ...
│   │   │   │   ├── train_split1_mean_duration.txt
│   │   │   │   ├── train_split2_mean_duration.txt
│   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── mapping.txt
│   │   ├── 50salads
│   │   │   ├── features
│   │   │   │   ├── ...
│   │   │   ├── groundTruth
│   │   │   │   ├── ...
│   │   │   ├── gt_arr
│   │   │   │   ├── ...
│   │   │   ├── gt_boundary_arr
│   │   │   │   ├── ...
│   │   │   ├── splits
│   │   │   │   ├── ...
│   │   │   ├── mapping.txt
│   │   ├── breakfast
│   │   │   ├── features
│   │   │   │   ├── ...
│   │   │   ├── groundTruth
│   │   │   │   ├── ...
│   │   │   ├── gt_arr
│   │   │   │   ├── ...
│   │   │   ├── gt_boundary_arr
│   │   │   │   ├── ...
│   │   │   ├── splits
│   │   │   │   ├── ...
│   │   │   ├── mapping.txt

```

For training and evaluating on GTEA, 50Salads and Breakfast, please refer to [Training and Test Tutorial](/docs/en/user_guides/train_test.md).
