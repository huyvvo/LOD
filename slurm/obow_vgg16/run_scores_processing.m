clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
POOL_SIZE=36;
PROPSET_NAME='coco_train_20k_vgg16_bn_obow80_cnn_05_20_10_05_10_with_roots_widemargin_2000';
SCORE_NAME='vgg16_bn_obow80_relu53_77_roi_pooling_noresize_01_symmetric_100_neighbor_cos_vgg16_bn_obow80_relu53_resize_normalized_1000';
num_chunks = 36;
DATA_ROOT = fullfile(LOD_ROOT, 'data');

if ~isempty(gcp('nocreate'))
  delete(gcp('nocreate'));
end
parpool(POOL_SIZE);

parfor pidx = 1:num_chunks
  gather_matrix_parts_SSMC_symmetric(PROPSET_NAME, num_chunks, 1, pidx, SCORE_NAME, 'DATA_ROOT', DATA_ROOT);
end
gather_norm_factor(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/confidence_symmetric_imdb_matrix_SSMC_norm_factor', SCORE_NAME));
