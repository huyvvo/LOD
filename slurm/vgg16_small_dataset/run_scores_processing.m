clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
PROPSET_NAME='voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000';
SCORE_NAME='vgg16_relu53_77_roi_pooling_noresize_01_symmetric_10_neighbor_cos_vgg16_fc6_resize_normalized_1000';
num_chunks = 4;
DATA_ROOT = fullfile(LOD_ROOT, 'data');

for pidx = 1:num_chunks
  gather_matrix_parts_SSMC_symmetric(PROPSET_NAME, num_chunks, 1, pidx, SCORE_NAME, 'DATA_ROOT', DATA_ROOT);
end
gather_norm_factor(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/confidence_symmetric_imdb_matrix_SSMC_norm_factor', SCORE_NAME));
