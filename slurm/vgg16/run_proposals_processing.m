clear; clc
PWD = pwd();
cd ../../
setup;
cd(PWD)

% Modify these parameters for a new dataset
DATASET_NAME = 'coco_train_20k';
PROPSET = [DATASET_NAME, '_vgg16_cnn_03_20_10_05_10_with_roots'];
DATA_ROOT = fullfile(LOD_ROOT, 'data');

% Gather jobs' outputs into a 'mixed_all_lite.mat' file
compress_proposals(PROPSET,'vgg16_relu43_lite',1,'DATA_ROOT', DATA_ROOT);
compress_proposals(PROPSET,'vgg16_relu53_lite',1,'DATA_ROOT', DATA_ROOT);
combine_proposals(PROPSET, {'vgg16_relu43_lite', 'vgg16_relu53_lite'}, 1, 'DATA_ROOT', DATA_ROOT, 'verbose', false);

% Choose maximum 2000 proposals
imdb = load(fullfile(DATA_ROOT, PROPSET, 'mixed/mixed_all_lite.mat'));
imdb = subsample_proposals(imdb, 2000);
imdb.num_props = cellfun(@(el) size(el,1), imdb.proposals);
PROPSET2000 = [PROPSET, '_2000'];
mkdir(fullfile(DATA_ROOT, PROPSET2000, 'mixed'))
savefile(fullfile(DATA_ROOT, PROPSET2000, 'mixed/mixed_lite.mat'), imdb);
PL = ProposalLoader(fullfile(DATA_ROOT, PROPSET2000, 'mixed/mixed_proposals'), numel(imdb.proposals), 2000);
PL.ProposalLoader_write_chunk(imdb.proposals);
