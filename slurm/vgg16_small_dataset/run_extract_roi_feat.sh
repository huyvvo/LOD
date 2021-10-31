# Modify these parameters for other dataset or if you want to use a different number of CPUs.
PROPSET_NAME=voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000
DATASET_NAME=voc60
NUM_IMAGES=60

# Launch slurm jobs
LOD_ROOT=$(readlink -f ../../)

cd "$LOD_ROOT"/main_model/feature_extractor/vgg16/
python script_extract_roi_features.py \
       --cuda true -cl 0 \
       --START 0 --END "$NUM_IMAGES" \
       --home "$LOD_ROOT"/"data" \
       --propset "$PROPSET_NAME" \
       --propset_suffix "_lite" \
       --propset_h5py "false" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths" \
       --layer_name "relu53" \
       --deepfeat_name "vgg16_relu53_77_roi_pooling_noresize"