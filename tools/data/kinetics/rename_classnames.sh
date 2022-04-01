#!/usr/bin/env bash

# Rename classname for convenience
DATASET=$1
if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
        echo "We are processing $DATASET"
else
        echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
        exit 0
fi

cd ../../../data/${DATASET}/
ls ./videos_train | while read class; do \
  newclass=`echo $class | tr " " "_" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "videos_train/${class}" "videos_train/${newclass}";
  fi
done

ls ./videos_val | while read class; do \
  newclass=`echo $class | tr " " "_" `;
  if [ "${class}" != "${newclass}" ]
  then
    mv "videos_val/${class}" "videos_val/${newclass}";
  fi
done

cd ../../tools/data/kinetics/
