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
LINE_NUMBER_MATCHING=$(sed -n '/## Inference with Pre-Trained Models/=' getting_started.md) && sed -i "$((${LINE_NUMBER_MATCHING} - 1)) r prepare_data.md" getting_started.md

sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' recognition_models.md
sed -i 's=](/=](http://gitlab.sz.sensetime.com/open-mmlab/mmaction-lite/tree/master/=g' localization_models.md
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
