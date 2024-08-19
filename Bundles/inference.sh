#!/bin/bash

BUNDLE="/opt/AutoPET_PET_Classification"
PYTHONPATH=$BUNDLE

/opt/conda/envs/MONAI/bin/python -m monai.bundle run inference \
    --bundle_root "$BUNDLE" \
    --data_dir "/input" \
    --output-dir "/output" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file  "['$BUNDLE/configs/DenseNet.yaml','$BUNDLE/override.yaml','$BUNDLE/configs/inference.yaml']"

/opt/conda/envs/MONAI/bin/python /opt/run_segmentation.py