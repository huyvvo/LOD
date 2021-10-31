clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
POOL_SIZE = 36;
PROPSET_NAME = 'coco_train_20k_vgg16_bn_obow80_cnn_05_20_10_05_10_with_roots_widemargin_2000';
SCORE_NAME = 'vgg16_bn_obow80_relu53_77_roi_pooling_noresize_01_symmetric_100_neighbor_cos_vgg16_bn_obow80_relu53_resize_normalized_1000';
DATA_ROOT = fullfile(LOD_ROOT, 'data');


if ~isempty(gcp('nocreate'))
  delete(gcp('nocreate'));
end
parpool(POOL_SIZE);
QP_large_SSMC(PROPSET_NAME, SCORE_NAME, POOL_SIZE, 1, 'DATA_ROOT', DATA_ROOT, 'save_step', 50);
