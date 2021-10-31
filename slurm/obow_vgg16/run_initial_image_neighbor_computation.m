clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
POOL_SIZE = 36; % Number of CPUs to run Matlab in parallel
DATASET_NAME = 'coco_train_20k';
DATA_ROOT = fullfile(LOD_ROOT, 'data');
NUM_IMAGES = 19817;


if ~isempty(gcp('nocreate'))
  delete(gcp('nocreate'));
end
parpool(POOL_SIZE);

gather_image_features(DATASET_NAME,1,'vgg16_bn_obow80_relu53_resize','DATA_ROOT', DATA_ROOT);
ints = arrayfun(@(i) [round(NUM_IMAGES/POOL_SIZE*i+1),round(NUM_IMAGES/POOL_SIZE*(i+1))], 0:POOL_SIZE-1, 'Uni', false);
parfor pidx = 1:POOL_SIZE
  a = ints{pidx}(1);
  b = ints{pidx}(2);
  compute_neighbor_cnn_large_scale(DATASET_NAME,'vgg16_bn_obow80_relu53_resize',1,[a:b],'DATA_ROOT',DATA_ROOT);
end
gather_neighbor_large_scale(DATASET_NAME,1,'neighbor_cos_vgg16_bn_obow80_relu53_resize',100, 'DATA_ROOT', DATA_ROOT);
fprintf('Number of score blocks to compute is %d\n', ...
		compute_num_rows_symmetric(fullfile(LOD_ROOT, 'data', DATASET_NAME, ...
								   'neighbor_cos_vgg16_bn_obow80_relu53_resize/mixed/100.mat')));
