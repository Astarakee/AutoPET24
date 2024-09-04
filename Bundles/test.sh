#!/bin/bash

# Capture the start time
start_time=$(date +%s)

SCRIPTPATH="$(dirname "$( cd "$(dirname "$0")" ; pwd -P )")"
SCRIPTPATHCURR="$( cd "$(dirname "$0")" ; pwd -P )"
echo $SCRIPTPATH

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet_baseline-output-$VOLUME_SUFFIX

echo "Volume created, running evaluation"
# Do not change any of the parameters to docker run, these are fixed
# --gpus="device=0" \
docker run -it --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="2g" \
        --pids-limit="256" \
        --gpus="all" \
        -v /home/simben/Desktop/AutoPET_24/input/:/input/ \
        -v /home/simben/Desktop/AutoPET_24/tmp/:/opt/tmp/ \
        -v /home/simben/Desktop/AutoPET_24/output/:/output/ \
        autopet_classification:v1
        #-v autopet_baseline-output-$VOLUME_SUFFIX:/output/ \
echo "Evaluation done"

docker volume rm autopet_baseline-output-$VOLUME_SUFFIX

# Capture the end time and print difference
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Total runtime: $elapsed_time seconds"