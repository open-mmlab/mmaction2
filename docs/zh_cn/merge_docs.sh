#!/usr/bin/env bash

## gather models
cat  ../../configs/localization/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Localization Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" > localization_models.md
cat  ../../configs/recognition/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Action Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" > recognition_models.md
cat  ../../configs/recognition_audio/*/README.md | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" >> recognition_models.md
cat  ../../configs/detection/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Spatio Temporal Action Detection Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" > detection_models.md
cat  ../../configs/skeleton/*/README.md  | sed "s/md#t/html#t/g" | sed "s/#/#&/" | sed '1i\# Skeleton-based Action Recognition Models' | sed 's/](\/docs\//](/g' | sed 's=](/=](https://github.com/open-mmlab/mmaction2/tree/master/=g' | sed "s/getting_started.html##t/getting_started.html#t/g" > skeleton_models.md
