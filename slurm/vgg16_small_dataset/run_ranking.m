clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
POOL_SIZE = 4;
PROPSET_NAME = 'voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000';
SCORE_NAME = 'vgg16_relu53_77_roi_pooling_noresize_01_symmetric_10_neighbor_cos_vgg16_fc6_resize_normalized_1000';
DATA_ROOT = fullfile(LOD_ROOT, 'data');


if ~isempty(gcp('nocreate'))
  delete(gcp('nocreate'));
end
parpool(POOL_SIZE);
QP_large_SSMC(PROPSET_NAME, SCORE_NAME, POOL_SIZE, 1, 'DATA_ROOT', DATA_ROOT, 'save_step', 50);
