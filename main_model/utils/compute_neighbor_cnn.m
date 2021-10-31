function [] = compute_neighbor_cnn(imgset, feat_name, class_indices, varargin)
% Compute k nearest neighbors from CNN features.
%
% [] = compute_neighbor_cnn(imgset, feat_name, class_indices, varargin)
%
% Default arguments:
% DATA_ROOT = '~/data';
% similarity_measure = 'cos';
% num_neighbors = 100;

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
  fprintf('Loading features ');
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
  feats = cell2mat(feats);

  if strcmp(similarity_measure, 'cos')
    norm_feats = feats ./ sqrt(sum(feats.*feats, 2));
    similarity = norm_feats * norm_feats';
    similarity(sub2ind([n,n], 1:n, 1:n)) = -1;
  elseif strcmp(similarity_measure, 'l2')
    norm_2 = sum(feats .* feats, 2);
    DIST = repmat(norm_2, 1, n) + repmat(norm_2', n, 1) - 2*feats*feats';
    similarity = (1 + max(DIST(:))) - DIST;
    similarity(sub2ind([n,n], 1:n, 1:n)) = -1;
  end
  
  e = cell(n,1);
  for i = 1:n 
    [~, max_idx] = sort(similarity(i,:), 'descend');
    e{i} = max_idx(1:min(end-1, num_neighbors));
  end

  save_path = fullfile(root, sprintf('neighbor_%s_%s', similarity_measure, feat_name), clname);
  mkdir(save_path);
  fprintf('Saving to %s\n', save_path);
  save(fullfile(save_path, sprintf('%d.mat', num_neighbors)), 'e');

end