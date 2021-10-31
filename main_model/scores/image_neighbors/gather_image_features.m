function [] = gather_image_features(imgset, class_indices, feat_name, varargin)
%
% [] = gather_image_features(imgset, class_indices, feat_name, varargin)
%
% Default arguments:
% DATA_ROOT = '~/data';
% normalize_score = true;
% similarity_measure = 'cos';

% Default arguments
DATA_ROOT = '~/data';
normalize_score = true;
similarity_measure = 'cos';

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'similarity_measure', 'normalize_score'};
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
fprintf('class indices: %d ', class_indices); fprintf('\n');
fprintf('feat_name: %s\n', feat_name);
fprintf('Default arguments: \n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('normalize_score: %s\n', string(normalize_score));
fprintf('similarity_measure: %s\n', similarity_measure);

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
  if isempty(gcp('nocreate'))
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
  else
    parfor i = 1:n 
      if mod(i, 100) == 1
        fprintf('%d ', i);
      end
      feat = getfield(load(fullfile(root, clname, 'features/image', feat_name, sprintf('%d.mat', i))), 'data');
      if size(feat,1) == 1
        feat = feat';
      end
      feats{i} = mean(reshape(feat, size(feat,1), []), 2)';
    end
  end
  fprintf('\n');
  fprintf('Features loaded in %.2f\n', toc(load_tic));
  feats = cell2mat(feats);
  save_data_par(fullfile(root, clname, 'features/image', feat_name, 'features.mat'), feats, 'data');

  if normalize_score
    if strcmp(similarity_measure, 'cos')
      feats = feats ./ sqrt(sum(feats.^2,2));
    else
      error(sprintf('Similarity %s not supported!', similarity_measure)); 
    end
  end
  save_data_par(fullfile(root, clname, 'features/image', feat_name, sprintf('features_%s.mat', similarity_measure)), feats, 'data');  

end