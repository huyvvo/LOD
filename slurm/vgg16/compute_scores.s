#!/bin/bash -e
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=2200MB
#SBATCH --job-name=compute_scores_vgg16_%a
#SBATCH --output=logs/%A_%a

echo "SLURM_JOB_ID" $SLURM_JOB_ID

matlab -nodisplay -nosplash -r \
"compute_scores_script("$SLURM_ARRAY_TASK_ID", "$(printf "'%s'" $LOD_ROOT)", "$STEP", "$NUM_BLOCKS", "$(printf "'%s'" $DATASET_NAME)", "$(printf "'%s'" $PROPSET_NAME)");exit"
