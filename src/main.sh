#!/bin/bash

ROOT_DIR=$(pwd)

# preparing directories
mkdir -p $data_dir $log_dir/train $log_dir/test


# Download Dataset and Model
cd $(pwd)/snn_angular_velocity
FILE=pretrained/cnn5-avgp-fc1.pt
if [ ! -f "$FILE" ]; then
    wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/models/pretrained.pt" -O $FILE
fi

# Download and extract test dataset.
cd $data_dir

FILE=$data_dir/test.tar.zst
if [ ! -f "$FILE" ]; then
    wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/test.tar.zst" -O $FILE && \
    zstd -vd test.tar.zst && \
    tar -xvf test.tar
fi

# FILE=$data_dir/train.tar.zst
# if [ ! -f "$FILE" ]; then
#     wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/train.tar.zst" -O $FILE && \
#     zstd -vd train.tar.zst && \
#     tar -xvf train.tar
# fi

FILE=$data_dir/val.tar.zst
if [ ! -f "$FILE" ]; then
    wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/val.tar.zst" -O $FILE && \
    zstd -vd val.tar.zst && \
    tar -xvf val.tar
fi

# wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/imgs.tar" -O $data_dir/imgs.tar && \
# cd $data_dir && \
# zstd -vd imgs.tar.zst && \
# tar -xvf imgs.tar && \
# rm imgs.*

cd ${ROOT_DIR}/snn_angular_velocity
python3.7 train.py --write && \
python3.7 test.py --write
