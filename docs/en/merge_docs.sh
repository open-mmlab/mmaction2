#!/usr/bin/env bash

# gather models
mkdir -p model_zoo
cat  ../../configs/localization/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Localization Models' | sed 's/](\/docs\/en/](../g' |sed "s/getting_started.html##t/getting_started.html#t/g" > model_zoo/localization_models.md
cat  ../../configs/recognition/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Recognition Models' | sed 's/](\/docs\/en/](../g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/recognition_models.md
cat  ../../configs/recognition_audio/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed 's/](\/docs\/en/](../g' | sed "s/getting_started.html##t/getting_started.html#t/g" >>  model_zoo/recognition_models.md
cat  ../../configs/detection/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Spatio Temporal Action Detection Models' | sed 's/](\/docs\/en/](../g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/detection_models.md
cat  ../../configs/skeleton/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Skeleton-based Action Recognition Models' | sed 's/](\/docs\/en/](../g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/skeleton_models.md

# gather projects
# TODO: generate table of contents for project zoo
cat ../../projects/README.md > projectzoo.md
cat ../../projects/*/README.md >> projectzoo.md

# gather datasets
cat  ../../tools/data/*/README.md | sed 's/# Preparing/# /g' | sed 's/#/#&/' > prepare_data.md

sed -i 's/(\/tools\/data\/activitynet\/README.md/(#activitynet/g' supported_datasets.md
sed -i 's/(\/tools\/data\/kinetics\/README.md/(#kinetics-400600700/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mit\/README.md/(#moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/mmit\/README.md/(#multi-moments-in-time/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv1\/README.md/(#something-something-v1/g' supported_datasets.md
sed -i 's/(\/tools\/data\/sthv2\/README.md/(#something-something-v2/g' supported_datasets.md
sed -i "s/(\/tools\/data\/thumos14\/README.md/(#thumos14/g" supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101\/README.md/(#ucf-101/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ucf101_24\/README.md/(#ucf101-24/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jhmdb\/README.md/(#jhmdb/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hvu\/README.md/(#hvu/g' supported_datasets.md
sed -i 's/(\/tools\/data\/hmdb51\/README.md/(#hmdb51/g' supported_datasets.md
sed -i 's/(\/tools\/data\/jester\/README.md/(#jester/g' supported_datasets.md
sed -i 's/(\/tools\/data\/ava\/README.md/(#ava/g' supported_datasets.md
sed -i 's/(\/tools\/data\/gym\/README.md/(#gym/g' supported_datasets.md
sed -i 's/(\/tools\/data\/omnisource\/README.md/(#omnisource/g' supported_datasets.md
sed -i 's/(\/tools\/data\/diving48\/README.md/(#diving48/g' supported_datasets.md
sed -i 's/(\/tools\/data\/skeleton\/README.md/(#skeleton-dataset/g' supported_datasets.md

cat prepare_data.md >> supported_datasets.md

sed -i 's/](\/docs\/en\//](g' supported_datasets.md
sed -i 's/](\/docs\/en\//](g' changelog.md

sed -i 's/](\/docs\/en\//](..g' ./get_stated/*.md
sed -i 's/](\/docs\/en\//](..g' ./tutorials/*.md
