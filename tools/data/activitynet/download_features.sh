DATA_DIR="../../../data/ActivityNet/activitynet_feature_cuhk/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ISemndlSDS2FtqQOKL0t3Cjj9yk2yznF" -O "csv_mean_100.zip" && rm -rf /tmp/cookies.txt

unzip csv_mean_100.zip -d ${DATA_DIR}/
rm csv_mean_100.zip
