#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=2500MB
#SBATCH --job-name=extract_roi_feat_vgg16_%a
#SBATCH --output=logs/%A_%a


source ~/.bashrc
conda activate LOD
echo "SLURM_JOB_ID" $SLURM_JOB_ID

START=$((SLURM_ARRAY_TASK_ID*STEP-STEP))
END=$((SLURM_ARRAY_TASK_ID*STEP))

cd "$LOD_ROOT"/main_model/feature_extractor/obow_vgg16/
python script_extract_roi_features.py \
       --cuda false -cl 0 \
       --START "$START" --END "$END" \
       --home "$LOD_ROOT"/"data" \
       --propset "$PROPSET_NAME" \
       --propset_suffix "_lite" \
       --propset_h5py "false" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths" \
       --layer_name "relu53" \
       --deepfeat_name "vgg16_bn_obow80_relu53_77_roi_pooling_noresize"

