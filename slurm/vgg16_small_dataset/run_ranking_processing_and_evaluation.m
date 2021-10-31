clear; clc;
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
POOL_SIZE=4;
PROPSET_NAME = 'voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000';
SCORE_NAME = 'vgg16_relu53_77_roi_pooling_noresize_01_symmetric_10_neighbor_cos_vgg16_fc6_resize_normalized_1000';
DATA_ROOT = fullfile(LOD_ROOT, 'data');

if ~isempty(gcp('nocreate'))
  delete(gcp('nocreate'));
end
parpool(POOL_SIZE);

% Select objects from importance scores
% The indices of the selected regions in all images are saved to 
% fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/lsuod_results', ...
%           fullfile('em_ppr_SSMC/confidence_symmetric_imdb_matrix_SSMC', 
%					 sprintf('x_group_nms_0.30_tele_0.00010_alpha_0.10000_R1_1_coef_0.00010_%s.mat', SCORE_NAME)
v2x(PROPSET_NAME,1, fullfile('em_ppr_SSMC/confidence_symmetric_imdb_matrix_SSMC', sprintf('v_tele_0.00010_alpha_0.10000_R1_1_coef_0.00010_%s.mat', SCORE_NAME)),'DATA_ROOT', DATA_ROOT);

% Evaluate CorLoc, AP50, AP@[50:95]
% Load proposal sets
imdb = load(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/mixed_lite.mat'));
% Load object indices
% imdb.proposals{i}{x{i}, :} are objects selected by LOD.
x = load(fullfile(DATA_ROOT, PROPSET_NAME, 'mixed/lsuod_results', ...
         fullfile('em_ppr_SSMC/confidence_symmetric_imdb_matrix_SSMC', sprintf('x_group_nms_0.30_tele_0.00010_alpha_0.10000_R1_1_coef_0.00010_%s.mat', SCORE_NAME))));
% Compute scores
ap = arrayfun(@(el) x2pr(imdb.proposals, imdb.bboxes, x.x, el), 0.5:0.05:0.95);
[~,pr] = x2pr(imdb.proposals, imdb.bboxes, x.x, 0.5);
fprintf('CorLoc: %s, AP50: %s, AP@[50:95]: %s\n', string(pr(1)*100), string(ap(1)*100), string(mean(ap)*100));
