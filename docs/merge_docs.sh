#!/usr/bin/env bash

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md

sed -i 's/](\/docs\//](/g' ./tutorials/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' ./tutorials/*.md

sed -i 's/](\/docs\//](/g' ../tools/data/*/*.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' ../tools/data/*/*.md
cat  ../tools/data/*/*.md > prepare_data.md

sed -i 's/.md](\/tools\/data\/activitynet\/preparing_activitynet.md/](#activitynet/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/kinetics400\/preparing_kinetics400.md/](#kinetics-400/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/mit\/preparing_mit.md/](#moments-in-time/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/mmit\/preparing_mmit.md/](#multi-moments-in-time/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/sthv1\/preparing_sthv1.md/](#something-something-v1/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/sthv2\/preparing_sthv2.md/](#something-something-v2/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/thumos14\/preparing_thumos14.md/](#thumos-14/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/ucf101\/preparing_ucf101.md/](#ucf-101/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/ucf101_24\/preparing_ucf101_24.md/](#ucf101-24/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/jhmdb\/preparing_jhmdb.md/](#jhmdb/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/hvu\/preparing_hvu.md/](#hvu/g' data_preparation.md
sed -i 's/.md](\/tools\/data\/hmdb51\/preparing_hmdb51.md/](#hmdb51/g' data_preparation.md


sed -i 's/#/##&/' localization_models.md
sed -i 's/#/##&/' recognition_models.md
sed -i 's/md###t/html#t/g' localization_models.md
sed -i 's/md###t/html#t/g' recognition_models.md

sed -i 's/# Preparing/# /g' prepare_data.md
sed -i 's/#/##&/' prepare_data.md

sed -i '1i\## Action Localization Models' localization_models.md
sed -i '1i\## Action Recognition Models' recognition_models.md

sed -i '1i\## Preparing Datasets' prepare_data.md
cat prepare_data.md >> data_preparation.md

sed -i 's/](\/docs\//](/g' recognition_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' localization_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' recognition_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' localization_models.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' benchmark.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' getting_started.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' install.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' config.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' changelog.md
