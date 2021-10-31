function [] = compute_neighbor_cnn_large_scale(imgset, feat_name, class_indices, row_indices, varargin)
% Compute k nearest neighbors from CNN features.
%
% [] = compute_neighbor_cnn_large_scale(imgset, feat_name, class_indices, varargin)
%
% Default arguments:
% DATA_ROOT = '~/data';
% similarity_measure = 'cos';
% num_neighbors = 100;
%

% Default arguments:
DATA_ROOT = '~/data';
similarity_measure = 'cos';
num_neighbors = 100;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'similarity_measure', 'num_neighbors'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for varidx = 1:numel(varnames)
  evalc(sprintf('%s=varvals{%d}',varnames{varidx}, varidx));
end

fprintf('Arguments:\n');
fprintf('imgset: %s\n', imgset);
fprintf('feat_name: %s\n', feat_name);
fprintf('class indices: %d ', class_indices); fprintf('\n');
fprintf('Default arguments: \n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('similarity_measure: %s\n', similarity_measure);
fprintf('num_neighbors: %d\n', num_neighbors);

%-----------------------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
  n = numel(imdb.bboxes);

  feats = cell(n,1);
  fprintf('Loading features ...\n');
  load_tic = tic;
  for i = 1:n 
    if mod(i, 100) == 1
      fprintf('%d ', i);
    end
    feat = getfield(load(fullfile(root, clname, 'features/image', feat_name, sprintf('%d.mat', i))), 'data');
    if size(feat,1) == 1
      feat = feat';
    end
    feats{i} = mean(reshape(feat, size(feat,1), []), 2)';
  end
  fprintf('\n');
  fprintf('Features loaded in %.2f\n', toc(load_tic));
  feats = cell2mat(feats);

  e = cell(n,1);
  neighbor_tic = tic;
  if strcmp(similarity_measure, 'cos')
    norm_feats = feats ./ sqrt(sum(feats.*feats, 2));
    fprintf('Computing neighbors ')
    for i = row_indices 
      if mod(i, 100) == 1
        fprintf('%d ', i);
      end
      current_sim = norm_feats(i,:) * norm_feats';
      current_sim(i) = -1;
      [~, max_idx] = sort(current_sim, 'descend');
      e{i} = max_idx(1:num_neighbors);
      if ismember(i, e{i})
        e{i} = setdiff(e{i}, i);
      end
    end
    fprintf('\n');
  elseif strcmp(similarity_measure, 'l2')
    norm_2 = sum(feats .* feats, 2);
    fprintf('Computing neighbors ')
    for i = row_indices 
      if mod(i, 100) == 1
        fprintf('%d ', i);
      end
      current_dist = norm_2(i) + norm_2' - 2*feats(i,:)*feats';
      current_sim = (1+max(current_dist)) - current_dist;
      current_sim(i) = -1;
      [~, max_idx] = sort(current_sim, 'descend');
      e{i} = max_idx(1:num_neighbors);
      if ismember(i, e{i})
        e{i} = setdiff(e{i}, i);
      end
    end
    fprintf('\n');
  end
  fprintf('Neighbors found in %.2f\n', toc(neighbor_tic));
  e = e(row_indices);
  
  save_dir = fullfile(root, sprintf('neighbor_%s_%s/%s/%d', similarity_measure, feat_name, clname, num_neighbors));
  mkdir(save_dir);
  save_path = fullfile(save_dir, sprintf('%d_%d.mat', min(row_indices), max(row_indices)));
  fprintf('Saving results to %s\n', save_path);
  save(save_path, 'e');

end