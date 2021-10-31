#/bin/bash

# Modify these parameters for other dataset or if you want to use a different number of CPUs.
DATASET_NAME=coco_train_20k
NUM_IMAGES=19817
NUM_CPUS=200
# Launch slurm jobs
LOD_ROOT="$(readlink -f ../../)"
echo $LOD_ROOT
mkdir -p logs
sbatch --array=1-"$NUM_CPUS" --export=LOD_ROOT="$LOD_ROOT",DATASET_NAME="$DATASET_NAME",STEP="$(((NUM_IMAGES+NUM_CPUS-1)/NUM_CPUS))" extract_image_feat.s