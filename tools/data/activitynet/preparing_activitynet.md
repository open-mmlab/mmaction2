# Preparing ActivityNet

For basic dataset information, please refer to the official [website](http://activity-net.org/).
For action detection, you can either use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation) or extract feature with mmaction2 (which has better performance).
We release both pipeline.
Before we start, please make sure that current working directory is `$MMACTION2/tools/data/activitynet/`.

## Step 1. Download Annotations
First of all, you can run the following script to download annotation files.
```shell
bash download_annotations.sh
```

## Option 1: Use the ActivityNet rescaled feature provided in this [repo](https://github.com/wzmsltw/BSN-boundary-sensitive-network#code-and-data-preparation)

### Step 2. Prepare Videos Features
Then, you can run the following script to download activitynet features.
```shell
bash download_features.sh
```

### Step 3. Process Annotation Files
Next, you can run the following script to process the downloaded annotation files for training and testing.
It first merges the two annotation files together and then seperates the annoations by `train`, `val` and `test`.

```shell
python process_annotations.py
```

## Option 2: Extract ActivityNet feature using MMAction2.

### Step 2. Prepare Videos.
Then, you can run the following script to prepare videos.
The codes are adapted from the [official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics). Note that this might take a long time.
Some videos in the ActivityNet dataset might be no longer available on YouTube, so that after video downloading, the downloading scripts update the annotation file to make sure every video in it exists.

```shell
bash download_videos.sh
```

###

## Final Step. Check Directory Structure

After the whole data pipeline for ActivityNet preparation,
you will get the features and annotation files.

In the context of the whole project (for ActivityNet only), the folder structure will look like:

```
mmaction2
├── mmaction
├── tools
├── configs
├── data
│   ├── ActivityNet
│   │   ├── anet_anno_{train,val,test,full}.json
│   │   ├── anet_anno_action.json
│   │   ├── video_info_new.csv
│   │   ├── activitynet_feature_cuhk
│   │   │   ├── csv_mean_100
│   │   │   │   ├── v___c8enCfzqw.csv
│   │   │   │   ├── v___dXUJsj3yo.csv
│   │   │   |   ├── ..
```

For training and evaluating on ActivityNet, please refer to [getting_started.md](/docs/getting_started.md).
