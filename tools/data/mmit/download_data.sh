DATA_DIR="../../../data/mmit/"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} does not exist. Creating";
  mkdir -p ${DATA_DIR}
fi

cd ${DATA_DIR}

wget -c https://www.dropbox.com/s/sz3yd1o0gf09amh/Multi_Moments_in_Time.zip?dl=0

unzip Multi_Moments_in_Time_Raw.zip
rm Multi_Moments_in_Time.zip

if [ ! -d "./annotations" ]; then
  mkdir ./annotations
fi

mv *.txt annotations && mv *.csv annotations

cd -
