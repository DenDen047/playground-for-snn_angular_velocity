FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update

RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update

# Install latest version of python3
RUN apt-get install -y python3.7
RUN apt-get install -y python3.7-dev python3-pip
RUN apt-get install -y build-essential
RUN apt-get install -y libblas-dev libatlas-base-dev

# Upgrade pip
RUN apt-get install -y git wget

# slayerPytorch
WORKDIR /work
RUN git clone https://github.com/uzh-rpg/snn_angular_velocity.git

# Bind python3.7 to python
RUN touch ~/.bash_aliases
RUN echo alias python='/usr/bin/python3.7' >> ~/.bash_aliases

# snn_angular_velocity
WORKDIR /work/snn_angular_velocity
RUN python3.7 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Data and Log Directory
ENV data_dir $(pwd)/data
ENV log_dir $(pwd)/logs
RUN mkdir -p $data_dir $log_dir/train $log_dir/test

WORKDIR /work/snn_angular_velocity/slayerpytorch
RUN python3.7 setup.py install

# Download Dataset and Model
WORKDIR /work/snn_angular_velocity
RUN wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/models/pretrained.pt" -O pretrained/cnn5-avgp-fc1.pt

WORKDIR /work/snn_angular_velocity
RUN wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/test.tar.zst" -O $data_dir/test.tar.zst
WORKDIR $data_dir
RUN zstd -vd test.tar.zst
RUN tar -xvf test.tar
RUN rm test.*

WORKDIR /work/snn_angular_velocity
RUN wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/train.tar.zst" -O $data_dir/train.tar.zst
WORKDIR $data_dir
RUN zstd -vd train.tar.zst
RUN tar -xvf train.tar
RUN rm train.*

WORKDIR /work/snn_angular_velocity
RUN wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/val.tar.zst" -O $data_dir/val.tar.zst
WORKDIR $data_dir
RUN zstd -vd val.tar.zst
RUN tar -xvf val.tar
RUN rm val.*

WORKDIR /work/snn_angular_velocity
RUN wget "http://rpg.ifi.uzh.ch/data/snn_angular_velocity/dataset/imgs.tar" -O $data_dir/imgs.tar
WORKDIR $data_dir
RUN zstd -vd imgs.tar.zst
RUN tar -xvf imgs.tar
RUN rm imgs.*

WORKDIR /work/snn_angular_velocity
RUN python3.7 test.py