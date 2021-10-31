NUM_BLOCKS="$1"
DATASET_NAME=coco_train_20k
PROPSET_NAME=coco_train_20k_vgg16_cnn_03_20_10_05_10_with_roots_2000
NUM_CPUS=1200

# Launch slurm jobs
LOD_ROOT=$(readlink -f ../../)
mkdir -p logs
sbatch --array=1-"$NUM_CPUS" \
--export=LOD_ROOT="$LOD_ROOT",DATASET_NAME="$DATASET_NAME",PROPSET_NAME="$PROPSET_NAME",STEP="$(((NUM_BLOCKS+NUM_CPUS-1)/NUM_CPUS))",NUM_BLOCKS="$NUM_BLOCKS",PATH="$PATH" \
compute_scores.s
