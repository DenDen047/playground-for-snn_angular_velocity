#!/bin/bash

# preparing directories
mkdir -p $data_dir $log_dir/train $log_dir/test


# Download Dataset and Model
cd ./snn_angular_velocity && \
wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/models/pretrained.pt" -O pretrained/cnn5-avgp-fc1.pt

# Download and extract test dataset.
wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/test.tar.zst" -O $data_dir/test.tar.zst && \
cd $data_dir && \
zstd -vd test.tar.zst && \
tar -xvf test.tar && \
rm test.*

wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/train.tar.zst" -O $data_dir/train.tar.zst && \
cd $data_dir && \
zstd -vd train.tar.zst && \
tar -xvf train.tar && \
rm train.*

wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/val.tar.zst" -O $data_dir/val.tar.zst && \
cd $data_dir && \
zstd -vd val.tar.zst && \
tar -xvf val.tar && \
rm val.*

wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/imgs.tar" -O $data_dir/imgs.tar && \
cd $data_dir && \
zstd -vd imgs.tar.zst && \
tar -xvf imgs.tar && \
rm imgs.*

cd ./snn_angular_velocity && \
python3.7 test.py