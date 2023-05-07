#!/usr/bin/env bash

# NOTE(Shih-Lun): retrain models with new data
python3 -u MLModels/auto_update.py \
    data_main \
    data_out \
    MLModels \
    SVD \
    False

python3 -u MLModels/auto_update.py \
    data_main \
    data_out \
    MLModels \
    baseline \
    True

# build 2 docker images
# new baseline image
export DOCKERFILE_PATH="./docker_baseline/Dockerfile"
export PROJECT_PATH="."
export TAG="m3:baseline"
docker build -f $DOCKERFILE_PATH -t $TAG $PROJECT_PATH 2>&1

# new svd image
export DOCKERFILE_PATH="./docker_svd/Dockerfile"
export TAG="m3:svd"
docker build -f $DOCKERFILE_PATH -t $TAG $PROJECT_PATH 2>&1

# shut down the baseline service and start the new baseline service
cd docker_baseline
docker-compose down 2>&1
docker-compose up -d 2>&1

# shut down the svd service and start the new svd service
cd ..
cd docker_svd
docker-compose down 2>&1
docker-compose up -d 2>&1

# return to root
cd ..
