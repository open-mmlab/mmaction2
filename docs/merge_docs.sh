#!/usr/bin/env bash

sed -i '$a\\n' ../demo/README.md

sed -i 's/(\/tools\/data\/activitynet\/README.md/(#activitynet/g' supported_datasets.md
sed -i 's/(\/tools\/data\/kinetics\/README.md/(#kinetics-400-600-700/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mit\/README.md/(#moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mmit\/README.md/(#multi-moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv1\/README.md/(#something-something-v1/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv2\/README.md/(#something-something-v2/g' supported_datasets.md
sed -i 's/(\/tools\/data\/thumos14\/README.md/(#thumos-14/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101\/README.md/(#ucf-101/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101_24\/README.md/(#ucf101-24/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jhmdb\/README.md/(#jhmdb/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hvu\/README.md/(#hvu/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hmdb51\/README.md/(#hmdb51/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jester\/README.md/(#jester/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ava\/README.md/(#ava/g' supported_datasets.md
sed -i 's/(\/tools\/data\/gym\/README.md/(#gym/g' supported_datasets.md

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md
cat  ../configs/recognition_audio/*/*.md >> recognition_models.md
cat  ../tools/data/*/*.md > prepare_data.md
cat  ../demo/README.md > demo.md

sed -i 's/#/##&/' localization_models.md
sed -i 's/#/##&/' recognition_models.md
sed -i 's/md###t/html#t/g' localization_models.md
sed -i 's/md###t/html#t/g' recognition_models.md
sed -i "s/md###t/html#t/g" demo.md

sed -i 's/# Preparing/# /g' prepare_data.md
sed -i 's/#/##&/' prepare_data.md

sed -i '1i\## Action Localization Models' localization_models.md
sed -i '1i\## Action Recognition Models' recognition_models.md

cat prepare_data.md >> supported_datasets.md

sed -i 's/](\/docs\//](/g' recognition_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' localization_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' recognition_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' localization_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' config.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' changelog.md
sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' ./tutorials/*.md
sed -i 's/](\/docs\//](/g' supported_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' supported_datasets.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' demo.md
