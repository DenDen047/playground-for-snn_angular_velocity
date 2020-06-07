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
RUN apt-get install -y git wget zstd

# env for slayerPytorch
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
ENV data_dir /data
ENV log_dir /logs

# install slayerpytorch
WORKDIR /work/snn_angular_velocity/slayerpytorch
RUN python3.7 setup.py install