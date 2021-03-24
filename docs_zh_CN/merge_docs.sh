#!/usr/bin/env bash

sed -i 's/(\/tools\/data\/activitynet\/README_zh-CN.md/(#activitynet/g' supported_datasets.md
sed -i 's/(\/tools\/data\/kinetics\/README_zh-CN.md/(#kinetics-400-600-700/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mit\/README_zh-CN.md/(#moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mmit\/README_zh-CN.md/(#multi-moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv1\/README_zh-CN.md/(#something-something-v1/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv2\/README_zh-CN.md/(#something-something-v2/g' supported_datasets.md
sed -i 's/(\/tools\/data\/thumos14\/README_zh-CN.md/(#thumos-14/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101\/README_zh-CN.md/(#ucf-101/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101_24\/README_zh-CN.md/(#ucf101-24/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jhmdb\/README_zh-CN.md/(#jhmdb/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hvu\/README_zh-CN.md/(#hvu/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hmdb51\/README_zh-CN.md/(#hmdb51/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jester\/README_zh-CN.md/(#jester/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ava\/README_zh-CN.md/(#ava/g' supported_datasets.md
sed -i 's/(\/tools\/data\/gym\/README_zh-CN.md/(#gym/g' supported_datasets.md

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md
cat  ../configs/recognition_audio/*/*.md >> recognition_models.md
cat  ../configs/detection/*/*.md > detection_models.md
cat  ../tools/data/*/README_zh-CN.md > prepare_data.md

sed -i 's/#/#&/' localization_models.md
sed -i 's/#/#&/' recognition_models.md
sed -i 's/#/#&/' detection_models.md
sed -i 's/md###t/html#t/g' localization_models.md
sed -i 's/md###t/html#t/g' recognition_models.md
sed -i 's/md###t/html#t/g' detection_models.md
sed -i "s/md###t/html#t/g" demo.md

sed -i 's/# 准备/# /g' prepare_data.md
sed -i 's/#/#&/' prepare_data.md

sed -i '1i\# 时序动作检测模型' localization_models.md
sed -i '1i\# 动作识别模型' recognition_models.md
sed -i '1i\# 时空动作检测模型' detection_models.md

cat prepare_data.md >> supported_datasets.md

sed -i 's/](\/docs\//](/g' recognition_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' localization_models.md
sed -i 's/](\/docs\//](/g' detection_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' recognition_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' localization_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' detection_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' changelog.md
sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' ./tutorials/*.md
sed -i 's/](\/docs\//](/g' supported_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' supported_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' demo.md
