## Getting started
LOD requires running multiple steps. This instruction will guide you through these steps and run LOD on a small subset of PASCAL VOC2007 dataset with **VGG16** features.

### 0. Download data
Download `mixed_lite.mat`, `mixed_image_paths_original.mat` and `images.zip` [here](https://drive.google.com/drive/folders/1XFT5jYQVe-jMmmUeLbLI0ztiiK0JzOc1?usp=sharing) and put them in `data/voc60/mixed`. Unzip `images.zip`. And run the following code in Matlab **from the `LOD` folder** to customize the image paths to your system.
```
setup;
imdb = load(fullfile(LOD_ROOT, 'data/voc60/mixed/mixed_image_paths_original.mat'));
imdb.image_paths = cellfun(@(el) fullfile(LOD_ROOT, el), imdb.image_paths, 'Uni', false);
savefile(fullfile(LOD_ROOT, 'data/voc60/mixed/mixed_image_paths.mat'), imdb);

```

**For all of the following steps, you are supposed to be in the `slurm/vgg16_small_dataset` folder.**

### 1. Image feature extraction
The first step of LOD involves extracting image features for all images in the dataset. The features will then be used to generate region proposals and compute image initial neighbors. In Bash, run:
```
# This script runs on a GPU
# Do not forget to 'conda activate LOD first'
bash run_extract_image_feat.sh
```
After all jobs are finished, you will have `NUM_IMAGES` files numbered from `1.mat` to `<NUM_IMAGES>.mat` in subfolders `vgg16_relu43`, `vgg16_relu53` and `vgg16_fc6_resize` of `data/voc60/mixed/features/image`.

### 2. Proposal generation
We use the method from [rOSD](https://github.com/huyvvo/rOSD) to generate proposals from CNN features. In Matlab, run:
```
create_proposals_script;
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
bash run_extract_roi_feat.sh
```
After all jobs are finished, you will have `NUM_IMAGES` files numbered from `1.mat` to `<NUM_IMAGES>.mat` in subfolder `vgg16_relu53_77_roi_pooling_noresize` of `data/voc60/mixed/features/proposals`.

### 5. Score computation
LOD uses [PHM](https://openaccess.thecvf.com/content_cvpr_2015/papers/Cho_Unsupervised_Object_Discovery_2015_CVPR_paper.pdf) score. To compute scores, in Matlab, run:
```
clear; clc; NUM_BLOCK=<Number of blocks at step 3 here>; compute_scores_script; run_scores_processing;
```

### 6. Ranking
In Matlab, run:
```
# Expected time: 45 minutes
run_ranking;
```
After the optimization is finished, the proposals' importance scores are saved in `data/voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000/mixed/lsuod_results/em_ppr_SSM`

### 7. Obtain objects and evaluation
In Matlab, run:
```
run_ranking_processing_and_evaluation;
```
### 8. Visualize results
In Matlab, run
```
run_visualization
```
then open `data/voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000/mixed/visualization/index.html` on a browser to see results.
