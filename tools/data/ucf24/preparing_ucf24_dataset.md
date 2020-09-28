# Preparing UCF-24

For basic dataset information, you can refer to the dataset [website](https://github.com/gurkirt/realtime-action-detection).
Before we start, please make sure that the directory is located at `$MMACTION2/tools/data/ucf24/`]

## Step 1. Download and Unzip the UCF-24 Dataset

First of all, you can run the following script to download the dataset from Google Driver and untar the dataset.

```shell
bash download_dataset.sh
```

This process may take a while, please be patient. There will be a success message output when the download is complete and when the unzip is complete.

## Step 2. Check Directory Structure

After Step 1, you will get the extracted RGB-images from videos, and you can also get "brox flow" and "real-time flow" optical flow images.

Of course, you will also get the dataset label. You will get two formats of label: `.txt`label and `.pkl`label.

- `.txt`label
> Txt files are in the folder `labels`. Each txt file records the action recognization and the location of the target in the corresponding frame.

- `.pkl`label
> Labels are in file `pyannot.pkl`. This file saves a dictionary data, which contains lots of sub-dictionary. Each sub-dictionary is the label of its corresponding video clip, which saves the action recognization of the clip, the number of frames, the vaild start and end frames, and the coordinate information of the target in each vaild frame.

In addition, the ucf24 data set provides a training set and test set division scheme, the division results are shown in the `trainlist01.txt` and `testlist01.txt`

In the context of the whole project (for UCF-24 only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ucf24
│   │   ├── rgb-images
│   │   │   ├── Basketball
│   │   │   │	├── v_Basketball_g01_c01
│   │   │   │	│	├── 00001.jpg
│   │   │   │	│	├── 00002.jpg
│   │   │   │	│	├── ...
│   │   │   │	│	├── 00140.jpg
│   │   │   │	│	├── 00141.jpg
│   │   │   ├── ...
│   │   │   ├── WalkingWithDog
│   │   │   │	├── v_WalkingWithDog_g01_c01
│   │   │   │	├── ...
│   │   │   │	├── v_WalkingWithDog_g25_c04
│   │   ├── brox-images
│   │   │   ├── ...
│   │   ├── fastOF-images
│   │   │   ├── ...
│   │   ├── labels
│   │   │   │	├── v_Basketball_g01_c01
│   │   │   │	│	├── 00009.txt
│   │   │   │	│	├── 00010.txt
│   │   │   │	│	├── ...
│   │   │   │	│	├── 00050.txt
│   │   │   │	│	├── 00051.txt
│   │   │   ├── ...
│   │   │   ├── WalkingWithDog
│   │   │   │	├── v_WalkingWithDog_g01_c01
│   │   │   │	├── ...
│   │   │   │	├── v_WalkingWithDog_g25_c04
│   │   ├── splitfiles
│   │   │   ├── trainlist01.txt
│   │   │   ├── testlist01.txt
│   │   │   ├── pyannot.pkl
```
