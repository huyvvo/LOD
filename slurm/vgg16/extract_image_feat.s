#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=extract_image_feat_vgg16_%a
#SBATCH --output=logs/%A_%a

source ~/.bashrc
conda activate LOD
echo "SLURM_JOB_ID" $SLURM_JOB_ID

START=$((SLURM_ARRAY_TASK_ID*STEP-STEP))
END=$((SLURM_ARRAY_TASK_ID*STEP))

cd "$LOD_ROOT"/main_model/feature_extractor/vgg16/

python script_extract_image_features.py \
       -l relu43 -rs false --cuda false -cl 0 \
       --START "$START" --END "$END" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"

python script_extract_image_features.py \
       -l relu53 -rs false --cuda false -cl 0 \
       --START "$START" --END "$END" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"

python script_extract_image_features.py \
       -l fc6 -rs True --cuda false -cl 0 \
       --START "$START" --END "$END" \
       --home "$LOD_ROOT"/"data" \
       --imgset "$DATASET_NAME" \
       --imgset_suffix "_image_paths"