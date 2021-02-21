DATA_DIR="../../../data/ActivityNet/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget https://raw.githubusercontent.com/wzmsltw/BSN-boundary-sensitive-network/master/data/activitynet_annotations/anet_anno_action.json

wget https://raw.githubusercontent.com/wzmsltw/BSN-boundary-sensitive-network/master/data/activitynet_annotations/video_info_new.csv

wget https://download.openmmlab.com/mmaction/localization/anet_activity_indexes_val.txt

cd -
