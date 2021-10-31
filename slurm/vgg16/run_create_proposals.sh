DATASET_NAME=coco_train_20k
NUM_IMAGES=19817
NUM_CPUS=200

# Launch slurm jobs
LOD_ROOT=$(readlink -f ../../)
mkdir -p logs
sbatch --array=1-"$NUM_CPUS" \
--export=LOD_ROOT="$LOD_ROOT",DATASET_NAME="$DATASET_NAME",STEP="$(((NUM_IMAGES+NUM_CPUS-1)/NUM_CPUS))",NUM_IMAGES="$NUM_IMAGES",PATH="$PATH" \
create_proposals.s
