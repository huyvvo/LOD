## Getting started
LOD requires running multiple steps. This instruction will guide you through these steps and run LOD on C20K dataset with VGG16 features. Please make sure that you prepare the necessary data by following [DATA.md](DATA.md).

We use [SLURM](https://slurm.schedmd.com/documentation.html) to launch distributed jobs. The launching slurm sbatch files and necessary scripts can be found in [`slurm`](../slurm/vgg16) folder.
To customize to another datasets or using a different number of CPUs/jobs, you can edit the parameters in these files. You might need to customize SLURM parameters (e.g., memory/time limit) in `.s` files to run on your own datasets. 

**For all of the following steps, you are supposed to be in the `slurm/vgg16` folder.**

### 1. Image feature extraction
The first step of LOD involves extracting image features for all images in the dataset. The features will then be used to generate region proposals and compute image initial neighbors. In Bash, run:
```
# The default number of CPUs/jobs is 200
# Expected time: 450 seconds / job
bash run_extract_image_feat.sh
```
After all jobs are finished, you will have `NUM_IMAGES` files numbered from `1.mat` to `<NUM_IMAGES>.mat` in subfolders `vgg16_relu43`, `vgg16_relu53` and `vgg16_fc6_resize` of `data/<DATASET_NAME>/mixed/features/image`.

### 2. Proposal generation
We use the method from [rOSD](https://github.com/huyvvo/rOSD) to generate proposals from CNN features. In Bash, run:
```
# The default number of CPUs/jobs is 200
# Expected time: 450 seconds / job
bash run_create_proposals.sh
```

After all jobs are finished, we gather their outputs to obtain the proposal set `coco_train_20k_vgg16_cnn_03_20_10_05_10_with_roots_2000`. In Matlab, run:
```
run_proposals_processing;
```

### 3. Initial image neighbor computation
In Matlab run:
```
run_initial_image_neighbor_computation;
```
This script will print the number of score blocks (`S_ij`) to compute. You should note this down. It will be a parameter to launch jobs in the `Score computation` step.

### 4. Region feature extraction
In Bash, run:
```
# The default number of CPUs/jobs is 200
# Expected time: 420 seconds / job
bash run_extract_roi_feat.sh
```
After all jobs are finished, you will have `NUM_IMAGES` files numbered from `1.mat` to `<NUM_IMAGES>.mat` in subfolder `vgg16_relu53_77_roi_pooling_noresize` of `data/<PROPSET_NAME>/mixed/features/proposals`.

### 5. Score computation
LOD uses [PHM](https://openaccess.thecvf.com/content_cvpr_2015/papers/Cho_Unsupervised_Object_Discovery_2015_CVPR_paper.pdf) score. To compute scores in parallel with slurm, in Bash, run:
```
# The default number of CPUs/jobs is 1200
# Expected time: 1 hour 30 minutes / job
bash run_compute_scores.sh <Number of blocks at step 3 here>
```
After all jobs are finished, in Matlab, run the following code to generate chunks of the score matrix:
```
run_scores_processing;
```

### 6. Ranking
In Matlab, run:
```
# Expected time: 45 minutes
run_ranking;
```
After the optimization is finished, the proposals' importance scores are saved in `data/coco_train_20k_vgg16_cnn_03_20_10_05_10_with_roots_2000/mixed/lsuod_results/em_ppr_SSMC`

### 7. Obtain objects and evaluation
In Matlab, run:
```
run_ranking_processing_and_evaluation;
```

