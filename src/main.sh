#!/bin/bash

PWD=$(pwd)
echo ${PWD}

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
    tar -xvf test.tar && \
    rm test.*
fi

FILE=$data_dir/train.tar.zst
if [ ! -f "$FILE" ]; then
    wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/train.tar.zst" -O $FILE && \
    zstd -vd train.tar.zst && \
    tar -xvf train.tar && \
    rm train.*
fi

FILE=$data_dir/val.tar.zst
if [ ! -f "$FILE" ]; then
    wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/val.tar.zst" -O $FILE && \
    zstd -vd val.tar.zst && \
    tar -xvf val.tar && \
    rm val.*
fi

# wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/imgs.tar" -O $data_dir/imgs.tar && \
# cd $data_dir && \
# zstd -vd imgs.tar.zst && \
# tar -xvf imgs.tar && \
# rm imgs.*

cd ${PWD}/snn_angular_velocity && \
python train.py && \
python test.py