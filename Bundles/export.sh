#!/bin/bash

./build.sh

docker save autopet_classification:v1 | gzip -c > /home/simben/Desktop/autopet_classification_v1.tar.gz