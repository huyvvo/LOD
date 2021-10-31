clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
PROPSET_NAME = 'voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000';
DATASET_NAME = 'voc60';
SCORE_NAME = 'vgg16_relu53_77_roi_pooling_noresize_01_symmetric_10_neighbor_cos_vgg16_fc6_resize_normalized_1000';
DATA_ROOT = fullfile(LOD_ROOT, 'data');

% Visualization
imdb = load(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/mixed_lite.mat'));
image_paths = load(fullfile(DATA_ROOT, DATASET_NAME, 'mixed/mixed_image_paths.mat'));
image_paths = image_paths.image_paths;
n = numel(imdb.bboxes);
imdb.images = arrayfun(@(i) imread(image_paths{i}), [1:n]', 'Uni', false);

x = load(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/lsuod_results', ...
         fullfile('em_ppr_SSMC/confidence_symmetric_imdb_matrix_SSMC', sprintf('x_group_nms_0.30_tele_0.00010_alpha_0.10000_R1_1_coef_0.00010_%s.mat', SCORE_NAME))));
x = arrayfun(@(i) x.x{i}(1:size(imdb.bboxes{i},1)), [1:n]', 'Uni', false);

save_path = fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/visualization');
mkdir(save_path);

visualize_result(imdb, save_path, x);