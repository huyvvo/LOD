#/bin/bash

# Modify these parameters for other dataset or if you want to use a different number of CPUs.
DATASET_NAME=voc60
NUM_IMAGES=60
LOD_ROOT="$(readlink -f ../../)"

cd "$LOD_ROOT"/main_model/feature_extractor/vgg16

python script_extract_image_features.py \
       -l relu43 -rs false --cuda true -cl 0 \
       --START 0 --END "$NUM_IMAGES" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"

python script_extract_image_features.py \
       -l relu53 -rs false --cuda true -cl 0 \
       --START 0 --END "$NUM_IMAGES" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"

python script_extract_image_features.py \
       -l fc6 -rs True --cuda true -cl 0 \
       --START 0 --END "$NUM_IMAGES" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"