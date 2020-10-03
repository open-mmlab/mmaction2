#!/usr/bin/env bash

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md
cat  ../tools/data/*/*.md > prepare_data.md

sed -i 's/#/##&/' localization_models.md
sed -i 's/#/##&/' recognition_models.md
sed -i 's/md###t/html#t/g' localization_models.md
sed -i 's/md###t/html#t/g' recognition_models.md

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
