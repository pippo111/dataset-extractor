#!/bin/sh
docker run -it --rm -v $(pwd):/usr/src/app -p 8888:8888 --user $(id -u):$(id -g) -e NB_UID=$(id -u) -e NB_GID=$(id -g) fbdev/ml-tf:latest bash
# jupyter notebook --ip=0.0.0.0 --port=8888
