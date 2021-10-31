function [] = compute_neighbor_cnn_large_scale(imgset, feat_name, class_indices, row_indices, varargin)
% Compute k nearest neighbors from CNN features.
%
% [] = compute_neighbor_cnn_large_scale(imgset, feat_name, class_indices, row_indices, varargin)
%
% Default arguments:
% DATA_ROOT = '~/data';
% similarity_measure = 'cos';
% num_neighbors = 100;
% feats = [];
%

% Default arguments:
DATA_ROOT = '~/data';
similarity_measure = 'cos';
num_neighbors = 100;
feats = [];

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'similarity_measure', 'num_neighbors', 'feats'};
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
fprintf('Feats empty: %s\n', string(isempty(feats)));

%-----------------------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']), 'bboxes');
  n = numel(imdb.bboxes);

  fprintf('Loading features ...\n');
  load_tic = tic;
  if ~isempty(feats)
    if strcmp(similarity_measure, 'cos')
      assert(all(abs(arrayfun(@(i) norm(feats(i,:)), 1:size(feats,1))-1) < 1e-4));
    end
  elseif exist(fullfile(root, clname, 'features/image', feat_name, sprintf('features_%s.mat', similarity_measure)), 'file') == 2
    feats = getfield(load(fullfile(root, clname, 'features/image', feat_name, sprintf('features_%s.mat', similarity_measure))), 'data');
  else
    feats = cell(n,1);
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
    if strcmp(similarity_measure, 'cos')
      feats = feats ./ sqrt(sum(feats.^2, 2));
    else 
      error(sprintf('Similarity %s not supported!', similarity_measure));
    end
  end
  fprintf('Features loaded in %.2f\n', toc(load_tic));

  e = cell(n,1);
  neighbor_tic = tic;
  fprintf('Computing neighbors ')
  S = feats(row_indices,:) * feats';
  for row = 1:numel(row_indices)
    current_sim = S(row,:);
    i = row_indices(row); 
    if mod(i, 100) == 1
      fprintf('%d ', i);
    end
    current_sim(i) = -1;
    [~, max_idx] = sort(current_sim, 'descend');
    e{i} = max_idx(1:num_neighbors);
    if ismember(i, e{i})
      e{i} = setdiff(e{i}, i);
    end
  end
  fprintf('\n');
  fprintf('Neighbors computed in %.2f\n', toc(neighbor_tic));
  e = e(row_indices);
  
  save_dir = fullfile(root, sprintf('neighbor_%s_%s/%s/%d', similarity_measure, feat_name, clname, num_neighbors));
  mkdir(save_dir);
  save_path = fullfile(save_dir, sprintf('%d_%d.mat', min(row_indices), max(row_indices)));
  fprintf('Saving results to %s\n', save_path);
  save(save_path, 'e');
  fprintf('Done!\n');
end