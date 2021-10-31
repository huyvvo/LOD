function [] = QP_large_SSMC(imgset, score_name, num_parts, class_indices, varargin)
%
% [] = QP_large_SSMC(imgset, score_name, num_parts, class_indices, varargin)
%
% Default parameters:
%
% DATA_ROOT = '~/data';
% score_type = 'confidence_symmetric_imdb_matrix_SSMC';
% norm_dir = ''; % [score_type '_norm_factor'] if empty
% scores_root = ''; % fullfile(DATA_ROOT, imgset) if empty
% save_dir_Q = 'lsuod_results/eigen_SSMC';
% save_dir_QP = 'lsuod_results/em_ppr_SSMC';
% num_iters_Q = 100;
% num_iters_P = 100;
% save_step = 5;
% R1 = 1;
% pseudo_gt_images_ratio = 0.1;
% tele = 0.0001;
% coef = 0.0001;
% early_stopping = true;


DATA_ROOT = '~/data';
score_type = 'confidence_symmetric_imdb_matrix_SSMC';
norm_dir = ''; % [score_type '_norm_factor'] if empty
scores_root = ''; % fullfile(DATA_ROOT, imgset) if empty
save_dir_Q = 'lsuod_results/eigen_SSMC';
save_dir_QP = 'lsuod_results/em_ppr_SSMC';
num_iters_Q = 100;
num_iters_P = 100;
save_step = 5;
R1 = 1;
pseudo_gt_images_ratio = 0.1;
tele = 0.0001;
coef = 0.0001;
early_stopping = true;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'score_type', 'norm_dir', 'scores_root', 'save_dir_Q', 'save_dir_QP', ...
              'num_iters_Q', 'num_iters_P', 'save_step', 'R1', 'tele', 'coef', 'early_stopping', 'pseudo_gt_images_ratio'};
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
fprintf('score_name: %s\n', score_name);
fprintf('num parts: %d\n', num_parts);
fprintf('class_indices: '); fprintf('%d ', class_indices); fprintf('\n');

fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('score_type: %s\n', score_type);
fprintf('norm_dir: %s\n', norm_dir);
fprintf('scores_root: %s\n', scores_root);
fprintf('save_dir_Q: %s\n', save_dir_Q);
fprintf('save_dir_QP: %s\n', save_dir_QP);
fprintf('tele: %f\n', tele);
fprintf('coef: %f\n', coef);
fprintf('num_iters_Q: %d\n', num_iters_Q);
fprintf('num_iters_P: %d\n', num_iters_P);
fprintf('save step: %d\n', save_step);
fprintf('early_stopping: %s\n', string(early_stopping));
fprintf('pseudo_gt_images_ratio: %f\n', pseudo_gt_images_ratio);
fprintf('R1: %f\n', R1);

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
  pivots = [0 cumsum(num_props')];
  
  save_path_Q = fullfile(root, clname, save_dir_Q, score_type);
  save_path_P = fullfile(root, clname, save_dir_QP, score_type);
  mkdir(fullfile(save_path_Q));
  mkdir(fullfile(save_path_P));

  norm_factor = getfield(load(fullfile(root, clname, norm_dir, score_name, 'norm_factor.mat')), 'norm_factor');
  norm_factor(norm_factor~=0) = 1 ./ norm_factor(norm_factor~=0);

  [image_edges, prop_edges] = get_part_info(num_props, num_parts);

  score_path = fullfile(scores_root, clname, score_type, score_name);

  %%%%%%%%%%%%%%%%%%%%%%
  % Q

  fprintf('Q iterations: ');
  Q_tic = tic;
  if exist(fullfile(save_path_Q, sprintf('v_coef_%.5f_%s.mat', coef, score_name)), 'file') ~= 2
    fprintf('Q solution is not found at at %s!\nComputing one.', ...
            fullfile(save_path_Q, sprintf('v_coef_%.5f_%s.mat', coef, score_name)));
    N_effective = nnz(norm_factor);
    E = zeros(N,1); E(norm_factor > 0) = 1;
    v = E/sqrt(N_effective);
    for iter = 1:num_iters_Q
      begin_iter = tic;
      new_v = right_matmul_par(v, score_path, num_parts, image_edges, prop_edges);
      new_v = (1-coef)*new_v + coef*sum(v)*E/N_effective;
      lambda = norm(new_v);
      new_v = new_v ./ lambda;
      norm_diff = norm(v-new_v);
      fprintf('Iter: %d, norm difference: %f, lambda: %s, %.2f secs ...\n', iter, norm_diff, string(lambda), toc(begin_iter));
      if norm_diff < 1e-6 & early_stopping
        v = new_v;
        break;
      else
        v = new_v;
      end
      if mod(iter,save_step) == 0
        save_data_par(fullfile(save_path_Q, sprintf('v_coef_%.5f_%s_iter_%d.mat', coef, score_name, iter)), v, 'v');
      end
    end
    save_data_par(fullfile(save_path_Q, sprintf('v_coef_%.5f_%s.mat', coef, score_name)), v, 'v');
  else 
    fprintf('Q found!\n');
    v = getfield(load(fullfile(save_path_Q, sprintf('v_coef_%.5f_%s.mat', coef, score_name))),'v');
  end
  fprintf('Q done in %.2f secs\n', toc(Q_tic));

  %%%%%%%%%%%%%%%%%%%%%%
  % P
  P_tic = tic;
  v = abs(v(:)');
  for pagerank_repeat = 1
    prior = get_personalized_vector(v, num_props, R1, pseudo_gt_images_ratio, pivots, n, N, norm_factor==0);
    v = prior;
    E = zeros(1,N); E(norm_factor > 0) = 1;
    fprintf('P iterations: \n');
    for iter = 1:num_iters_P
      begin_iter = tic;
      new_v = v .* norm_factor';
      new_v = left_matmul_par(new_v, score_path, num_parts, image_edges, prop_edges);
      new_v = (1-tele)*((1-coef)*new_v + coef*E/nnz(norm_factor)) + tele*prior;
      norm_diff = norm(v-new_v);
      fprintf('Iter: %d, norm difference: %2f, norm L1: %f, %.2f secs...\n', iter, norm_diff, sum(new_v), toc(begin_iter));
      if norm_diff < 1e-6 && early_stopping
        v = new_v;
        break;
      else 
        v = new_v;
      end
      if mod(iter, save_step) == 0
        save_data_par(fullfile(save_path_P, sprintf('v_tele_%.5f_alpha_%.5f_R1_%d_coef_%.5f_%s_iter_%d.mat', ...
                                                    tele, pseudo_gt_images_ratio, R1, coef, score_name, iter)), v', 'v');
      end
    end
  end
  save_data_par(fullfile(save_path_P, sprintf('v_tele_%.5f_alpha_%.5f_R1_%d_coef_%.5f_%s.mat', ...
                                          tele, pseudo_gt_images_ratio, R1, coef, score_name)), v', 'v');
  fprintf('P done in %.2f secs\n', toc(P_tic));
  toc(begin_time);
end

end

%--------------------------------------------------------------------------------------------

function [prior] = get_personalized_vector(v, num_props, R, ALPHA, pivots, n, N, is_isolated)
  v = mat2cell(v, 1, num_props);
  pseudo_objects = cell(1,n);
  for i = 1:n 
    [~, ids] = sort(v{i}, 'descend');
    pseudo_objects{i} = ids(1:min(numel(v{i}), R));
  end
  [~, valid_ids] = sort(cellfun(@max, v), 'descend');
  [pseudo_objects{valid_ids(round(n*ALPHA)+1:end)}] = deal([]);
  seed_ids = cell2mat(arrayfun(@(i) pseudo_objects{i} + pivots(i), 1:n, 'Uni', false));
  prior = zeros(1,N);
  prior(seed_ids) = 1;
  prior(is_isolated) = 0;

  K = nnz(prior);
  if K == 0 
    prior = zeros(1,N);
    prior(~is_isolated) = 1;
    prior = prior/nnz(prior);  
  else 
    prior = prior/K;
  end
end


function [new_v] = left_matmul_par(v, score_path, num_parts, image_edges, prop_edges)
  part_v = cell(1, num_parts);
  parfor part_idx = 1:num_parts

    img_first = image_edges{part_idx}(1); 
    img_last = image_edges{part_idx}(2);
    
    S = getfield(load(fullfile(score_path, sprintf('%d_%d.mat', img_first, img_last))), 'S');
    assert(S.is_row || S.is_symmetric);
    local_v = SSMC_left_mul(S, v);
    assert(numel(local_v) == prop_edges{part_idx}(2)-prop_edges{part_idx}(1)+1);
    part_v{part_idx} = local_v;
    S = [];
  end
  new_v = cell2mat(part_v);
end

function [new_v] = right_matmul_par(v, score_path, num_parts, image_edges, prop_edges)
  part_v = cell(num_parts,1);
  parfor part_idx = 1:num_parts
    
    img_first = image_edges{part_idx}(1);
    img_last = image_edges{part_idx}(2);
    
    S = getfield(load(fullfile(score_path, sprintf('%d_%d.mat', img_first, img_last))), 'S');
    local_v = SSMC_right_mul(S, v);
    assert(numel(local_v) == prop_edges{part_idx}(2)-prop_edges{part_idx}(1)+1);
    part_v{part_idx} = local_v;
    S = [];
  end
  new_v = cell2mat(part_v);
end
