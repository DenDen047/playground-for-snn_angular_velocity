#!/bin/bash

CURRENT_PATH=$(pwd)
IMAGE_NAME="denden047/snn_angular_velocity:latest"

docker build -t ${IMAGE_NAME} ./docker && \
docker run -it --rm \
    --gpus '"device=1"' \
    -v ${CURRENT_PATH}/src/:/src/ \
    -v ${CURRENT_PATH}/data/:/data/ \
    -v ${CURRENT_PATH}/logs/:/logs/ \
    -w /src \
    --ipc=host \
    ${IMAGE_NAME} \
    /bin/bash

# remove all <none> images
# docker rmi $(docker images -f "dangling=true" -q)