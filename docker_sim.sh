#!/bin/bash
docker container run --rm -u $(id -u):$(id -g) -v .:/data batsim-py python /data/test_final.py
