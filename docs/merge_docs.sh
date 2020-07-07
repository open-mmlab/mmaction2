#!/usr/bin/env bash

cat  ../configs/localization/*/*.md > localization_models.md
cat  ../configs/recognition/*/*.md > recognition_models.md

cat  ./tutorials/finetune.md ./tutorials/new_dataset.md ./tutorials/data_pipeline.md ./tutorials/new_modules.md > tutorials.md

cat  ../tools/data/*/*.md > prepare_data.md

sed -i 's/#/##&/' localization_models.md
sed -i 's/#/##&/' recognition_models.md

sed -i 's/#/#&/' tutorials.md

sed -i 's/# Preparing/# /g' prepare_data.md
sed -i 's/#/##&/' prepare_data.md

sed -i '1i\# Tutorials' tutorials.md

sed -i '1i\## Action Localization Models' localization_models.md
sed -i '1i\## Action Recognition Models' recognition_models.md

sed -i '1i\## Preparing Datasets' prepare_data.md
cat prepare_data.md >> data_preparation.md

sed -i 's/](\/docs\//](/g' recognition_models.md # remove /docs/ for link used in doc site
sed -i 's/](\/docs\//](/g' localization_models.md
sed -i 's/](\/docs\//](/g' data_preparation.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' recognition_models.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' localization_models.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' data_preparation.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' benchmark.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' getting_started.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' install.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' tutorials.md
sed -i 's/..\/imgs/_images/g' tutorials.md
sed -i 's/](new_dataset.md)/](#tutorial-2-adding-new-dataset)/g' tutorials.md

cat localization_models.md recognition_models.md > modelzoo.md
sed -i '1i\# Modelzoo' modelzoo.md

cat index.rst | grep -q "modelzoo.md"
if [ $? -ne 0 ] ;then
    sed -i '/api.rst/i\   modelzoo.md' index.rst
fi

cat index.rst | grep -q "tutorials.md"
if [ $? -ne 0 ] ;then
    sed -i '/api.rst/i\   tutorials.md' index.rst
fi
