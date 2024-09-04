#!/bin/bash

BUNDLE="/opt/AutoPET_PET_Classification"
PYTHONPATH=$BUNDLE:$PYTHONPATH:/opt

/opt/conda/envs/MONAI/bin/python /opt/convert_input_from_challenge_format.py convert-input-from-challenge-format --input-mha-folder /input --input-nifti-folder /opt/tmp/nifti_input
/opt/conda/envs/MONAI/bin/python /opt/convert_input_from_challenge_format.py preprocess-images --input-folder /input --preprocess-folder /opt/tmp/preprocessed

/opt/conda/envs/MONAI/bin/python -m monai.bundle run inference \
    --bundle_root "$BUNDLE" \
    --data_dir "/opt/tmp/nifti_input" \
    --output-dir "/opt/tmp/classification_output" \
    --meta_file "$BUNDLE/configs/metadata.json" \
    --config_file  "['$BUNDLE/configs/DenseNet.yaml','$BUNDLE/override.yaml','$BUNDLE/configs/inference.yaml']"

/opt/conda/envs/MONAI/bin/python /opt/run_segmentation.py --pet-tracer-prediction-table /opt/tmp/classification_output/PET_Tracer_predictions.csv --output-nifti-folder /opt/tmp/nifti_output --preprocess-folder /opt/tmp/preprocessed/imagesTs
/opt/conda/envs/MONAI/bin/python /opt/convert_input_from_challenge_format.py run-postprocessing --output-nifti-folder /opt/tmp/nifti_output --output-folder /output --preprocess-folder /opt/tmp/preprocessed
#/opt/conda/envs/MONAI/bin/python /opt/convert_input_from_challenge_format.py convert-output-to-challenge-format --output-mha-folder /output/images/automated-petct-lesion-segmentation --output-nifti-folder /opt/output