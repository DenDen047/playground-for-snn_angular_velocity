#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/snn_angular_velocity:latest"

docker build -t ${IMAGE_NAME} ./docker && \
docker rmi $(docker images -f "dangling=true" -q) && \
docker run -it --rm \
    --gpus '"device=0"' \
    -v ${CURRENT_PATH}/src/:/src/ \
    -w /src \
    --ipc=host \
    ${IMAGE_NAME} \
    /bin/bash