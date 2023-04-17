#!/usr/bin/env bash

# gather models
mkdir -p model_zoo
cat  ../../configs/localization/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Localization Models' | sed 's/](\/docs\/en/](../g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' |sed "s/getting_started.html##t/getting_started.html#t/g" > model_zoo/localization_models.md
cat  ../../configs/recognition/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Recognition Models' | sed 's/](\/docs\/en/](../g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/recognition_models.md
cat  ../../configs/recognition_audio/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed 's/](\/docs\/en/](../g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" >>  model_zoo/recognition_models.md
cat  ../../configs/detection/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Spatio Temporal Action Detection Models' | sed 's/](\/docs\/en/](../g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/detection_models.md
cat  ../../configs/skeleton/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Skeleton-based Action Recognition Models' | sed 's/](\/docs\/en/](../g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" >  model_zoo/skeleton_models.md

# gather projects
# TODO: generate table of contents for project zoo
cat ../../projects/README.md > projectzoo.md
cat ../../projects/example_project/README.md >> projectzoo.md
cat ../../projects/ctrgcn/README.md >> projectzoo.md
cat ../../projects/msg3d/README.md >> projectzoo.md

# gather datasets
cat supported_datasets.md > datasetzoo.md
cat  ../../tools/data/*/README.md | sed 's/# Preparing/# /g' | sed 's/#/#&/' >> datasetzoo.md

sed -i 's/(\/tools\/data\/activitynet\/README.md/(#activitynet/g' datasetzoo.md
sed -i 's/(\/tools\/data\/kinetics\/README.md/(#kinetics-400600700/g' datasetzoo.md
sed -i 's/(\/tools\/data\/mit\/README.md/(#moments-in-time/g' datasetzoo.md
sed -i 's/(\/tools\/data\/mmit\/README.md/(#multi-moments-in-time/g' datasetzoo.md
sed -i 's/(\/tools\/data\/sthv1\/README.md/(#something-something-v1/g' datasetzoo.md
sed -i 's/(\/tools\/data\/sthv2\/README.md/(#something-something-v2/g' datasetzoo.md
sed -i "s/(\/tools\/data\/thumos14\/README.md/(#thumos14/g" datasetzoo.md
sed -i 's/(\/tools\/data\/ucf101\/README.md/(#ucf-101/g' datasetzoo.md
sed -i 's/(\/tools\/data\/ucf101_24\/README.md/(#ucf101-24/g' datasetzoo.md
sed -i 's/(\/tools\/data\/jhmdb\/README.md/(#jhmdb/g' datasetzoo.md
sed -i 's/(\/tools\/data\/hvu\/README.md/(#hvu/g' datasetzoo.md
sed -i 's/(\/tools\/data\/hmdb51\/README.md/(#hmdb51/g' datasetzoo.md
sed -i 's/(\/tools\/data\/jester\/README.md/(#jester/g' datasetzoo.md
sed -i 's/(\/tools\/data\/ava\/README.md/(#ava/g' datasetzoo.md
sed -i 's/(\/tools\/data\/gym\/README.md/(#gym/g' datasetzoo.md
sed -i 's/(\/tools\/data\/omnisource\/README.md/(#omnisource/g' datasetzoo.md
sed -i 's/(\/tools\/data\/diving48\/README.md/(#diving48/g' datasetzoo.md
sed -i 's/(\/tools\/data\/skeleton\/README.md/(#skeleton-dataset/g' datasetzoo.md

cat prepare_data.md >> datasetzoo.md

sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' *.md
sed -i 's=](/=](https://github.com/open-mmlab/mmaction2/tree/main/=g' */*.md

sed -i 's/](\/docs\/en\//](g' datasetzoo.md
sed -i 's/](\/docs\/en\//](g' notes/changelog.md
sed -i 's/](\/docs\/en\//](..g' ./get_stated/*.md
sed -i 's/](\/docs\/en\//](..g' ./tutorials/*.md
