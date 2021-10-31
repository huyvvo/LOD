function [] = P_large_SSMC(imgset, score_name, num_parts, class_indices, varargin)
%
% [] = P_large_SSMC(imgset, score_name, num_parts, class_indices, varargin)
%
% Default parameters
% 
% DATA_ROOT = '~/data';
% score_type = 'confidence_symmetric_imdb_matrix_SSMC';
% scores_root = '';
% save_dir = 'lsuod_results/pagerank_SSMC';
% norm_dir = ''; % [score_type '_norm_factor'] if empty
% tele = 0.0001;
% num_iters = 100;
% save_step = 5;
% early_stopping = true;

DATA_ROOT = '~/data';
score_type = 'confidence_symmetric_imdb_matrix_SSMC';
scores_root = '';
save_dir = 'lsuod_results/pagerank_SSMC';
norm_dir = ''; % [score_type '_norm_factor'] if empty
tele = 0.0001;
num_iters = 100;
save_step = 5;
early_stopping = true;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'score_type', 'scores_root', 'save_dir', 'norm_dir', ...
              'tele', 'num_iters', 'save_step', 'early_stopping'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for varidx = 1:numel(varnames)
  evalc(sprintf('%s=varvals{%d}',varnames{varidx}, varidx));
end

if strcmp(scores_root, '')
  scores_root = fullfile(DATA_ROOT, imgset);
end

if strcmp(norm_dir, '')
  norm_dir = [score_type '_norm_factor'];
end

fprintf('Arguments:\n');
fprintf('imgset: %s\n', imgset);
fprintf('score_name: %s\n', score_name)
fprintf('num parts: %d\n', num_parts);
fprintf('class_indices: '); fprintf('%d ', class_indices); fprintf('\n');

fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('score_type: %s\n', score_type);
fprintf('scores_root: %s\n', scores_root);
fprintf('save_dir: %s\n', save_dir);
fprintf('norm_dir: %s\n', norm_dir);
fprintf('tele: %f\n', tele);
fprintf('num iters: %d\n', num_iters);
fprintf('save step: %d\n', save_step);
fprintf('early_stopping: %s\n', string(early_stopping));

%------------------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  
  begin_time = tic;
  clname = classes{cl};
  fprintf('Processing class %s\n', clname);
  num_props = getfield(load(fullfile(root, clname, [clname, '_lite.mat']),'num_props'),'num_props');
  N = sum(num_props);
  n = numel(num_props);

  save_path = fullfile(root, clname, save_dir, score_type);
  if exist(save_path, 'file') ~= 7
    mkdir(save_path);
  end

  norm_factor = getfield(load(fullfile(root, clname, norm_dir, score_name, 'norm_factor.mat')), 'norm_factor');
  norm_factor(norm_factor~=0) = 1 ./ norm_factor(norm_factor~=0);

  [image_edges, prop_edges] = get_part_info(num_props, num_parts);

  %-----------------------
  % PAGERANK
  E = zeros(1,N); E(norm_factor > 0) = 1;
  v = E/nnz(norm_factor);
  prior = v;
  score_path = fullfile(scores_root, clname, score_type, score_name);
  fprintf('P iterations: \n');
  for iter = 1:num_iters
    begin_iter = tic;
    new_v = v.*norm_factor';
    new_v = left_matmul_par(new_v, score_path, num_parts, image_edges, prop_edges);
    new_v = (1-tele)*new_v + tele*prior;
    fprintf('Iter: %d, norm difference: %f, norm L1: %f, %.2f secs...\n', iter, norm(new_v-v), sum(new_v), toc(begin_iter));
    if norm(v-new_v) < 1e-6 && early_stopping
      v = new_v;
      break;
    else 
      v = new_v;
    end
    if mod(iter, save_step) == 0
      save_data_par(fullfile(save_path, sprintf('v_tele_%.5f_%s_iter_%d.mat', tele, score_name, iter)), v', 'v');
    end
  end
  toc(begin_time);
  save_data_par(fullfile(save_path, sprintf('v_tele_%.5f_%s.mat', tele, score_name)), v', 'v');
end

end

%--------------------------------------------------------------------------------------------

function [new_v] = left_matmul_par(v, score_path, num_parts, image_edges, prop_edges)
  part_v = cell(1, num_parts);
  parfor part_idx = 1:num_parts

    img_first = image_edges{part_idx}(1); 
    img_last = image_edges{part_idx}(2);
    
    S = getfield(load(fullfile(score_path, sprintf('%d_%d.mat', img_first, img_last))), 'S');
    assert(~S.is_row || S.is_symmetric);
    local_v = SSMC_left_mul(S, v);
    assert(numel(local_v) == prop_edges{part_idx}(2)-prop_edges{part_idx}(1)+1);
    part_v{part_idx} = local_v;
    S = [];
  end
  new_v = cell2mat(part_v);
end