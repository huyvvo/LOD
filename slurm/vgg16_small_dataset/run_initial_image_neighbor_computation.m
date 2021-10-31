clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
DATASET_NAME = 'voc60';
DATA_ROOT = fullfile(LOD_ROOT, 'data');
NUM_IMAGES = 60;

compute_neighbor_cnn(DATASET_NAME,'vgg16_fc6_resize',1,'DATA_ROOT',DATA_ROOT, 'num_neighbors', 10);
fprintf('Number of score blocks to compute is %d\n', ...
		compute_num_rows_symmetric(fullfile(LOD_ROOT, 'data', DATASET_NAME, 'neighbor_cos_vgg16_fc6_resize/mixed/10.mat')));
