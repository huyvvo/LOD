#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=2800MB
#SBATCH --job-name=create_proposals_vgg16_%a
#SBATCH --output=logs/%A_%a

echo "SLURM_JOB_ID" $SLURM_JOB_ID

START=$((SLURM_ARRAY_TASK_ID*STEP-STEP+1))
END=$((SLURM_ARRAY_TASK_ID*STEP))

matlab -nodesktop -nosplash -r \
"create_proposals_script("$START", "$END", "$NUM_IMAGES", "$(printf "'%s'" $DATASET_NAME)", "$(printf "'%s'" $LOD_ROOT)"); exit"
