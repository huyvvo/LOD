DATASET_NAME='voc60';
PROPSET_NAME='voc60_vgg16_cnn_03_20_10_05_10_with_roots_2000';
CODE_ROOT = '../../';
DATA_ROOT = fullfile(CODE_ROOT, 'data');

script(1, NUM_BLOCKS, NUM_BLOCKS, DATASET_NAME, PROPSET_NAME, 'CODE_ROOT', CODE_ROOT, 'DATA_ROOT', DATA_ROOT);

function script(START, END , NUM_BLOCKS, DATASET_NAME, PROPSET_NAME, varargin)
%
% script(START, END, NUM_BLOCKS, varargin) 
%
% Default arguments:
% CODE_ROOT = 'LOD'
% DATA_ROOT = 'LOD/data'

% Default argumentss;
CODE_ROOT = 'LOD';
DATA_ROOT = 'LOD/data';

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'CODE_ROOT', 'DATA_ROOT'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for varidx = 1:numel(varnames)
  evalc(sprintf('%s=varvals{%d}',varnames{varidx}, varidx));
end

fprintf('START: %d, END: %d\n', START, END);
fprintf('CODE_ROOT: %s\n', CODE_ROOT);
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('DATASET_NAME: %s\n', DATASET_NAME);
fprintf('PROPSET_NAME: %s\n', PROPSET_NAME);
fprintf('NUM_BLOCKS: %d\n', NUM_BLOCKS);

%----------------------------------

PWD = pwd();
cd(CODE_ROOT);
setup;
cd(PWD);

END = min(END,NUM_BLOCKS);
if START > NUM_BLOCKS
  fprintf('START > NUM_BLOCKS, do nothing!\n'); 
  return;
end

row_indices = START:END;
fprintf('Row indices: %d to %d\n', row_indices(1), row_indices(end));

%--------------------------------------
% MAIN CODE GO HERE

class_indices = 1;
fprintf('Class indices: '); fprintf('%d ', class_indices); fprintf('\n');

args = struct;
all_classes = 1:1;
big_classes = [1];
%--------------------------------------------------------------------------------------

propset = PROPSET_NAME;
imgset = DATASET_NAME;
feat_root = fullfile(DATA_ROOT, propset);
small_imdb = false;
saveconf = true;
savestd = false;
prop_step = 2000;

[args.root{all_classes}] = deal(fullfile(DATA_ROOT, propset));
args.clname = get_classes(propset);
[args.feat_root{all_classes}] = deal(feat_root);
[args.small_imdb{all_classes}] = deal(small_imdb);
[args.saveconf{all_classes}] = deal(saveconf);
[args.savestd{all_classes}] = deal(savestd);
[args.prop_step{all_classes}] = deal(prop_step);

%-------------------------------
PHM_type ='';
symmetric = true;
num_pos = 1000;  
num_pos_text = '1000';
stdver = 4;
max_iter = 10000;
area_ratio = 0.5;
area_ratio_text = '05';

[args.PHM_type{all_classes}] = deal(PHM_type);
[args.symmetric{all_classes}] = deal(symmetric);
[args.num_pos{all_classes}] = deal(num_pos);
[args.num_pos_text{all_classes}] = deal(num_pos_text);
[args.stdver{all_classes}] = deal(stdver);
[args.max_iter{all_classes}] = deal(max_iter);
[args.area_ratio{all_classes}] = deal(area_ratio);
[args.area_ratio_text{all_classes}] = deal(area_ratio_text);

%-------------------------------
deepfeat = true;
feat_type='vgg16_relu53_77_roi_pooling_noresize';
sim_type = '01';

[args.deepfeat{all_classes}] = deal(deepfeat);
[args.feat_type{all_classes}] = deal(feat_type);
[args.sim_type{all_classes}] = deal(sim_type);

%-------------------------------

[args.prefiltered_nb{all_classes}] = deal(false);
[args.num_nb{all_classes}] = deal(0);
[args.nb_root{all_classes}] = deal({});
[args.nb_type{all_classes}] = deal({''});

prefiltered_nb = true;
num_nb = 10;
nb_root = {...
           fullfile(DATA_ROOT, imgset), ...
          };
nb_type = {...
           'neighbor_cos_vgg16_fc6_resize', ...
          };
indices_name = [];

[args.prefiltered_nb{big_classes}] = deal(prefiltered_nb);
[args.num_nb{big_classes}] = deal(num_nb);
[args.nb_root{big_classes}] = deal(nb_root);
[args.nb_type{big_classes}] = deal(nb_type);
[args.indices_name{big_classes}] = deal(indices_name);

%-------------------------------
compute_scores_large_scale_symmetric(args, class_indices, row_indices);
disp('done')


end
