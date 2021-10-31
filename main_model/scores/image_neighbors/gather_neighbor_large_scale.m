function [] = gather_neighbor_large_scale(imgset, class_indices, neighbor_name, num_neighbors, varargin)
%
% [] = gather_neighbor_large_scale(imgset, class_indices, neighbor_name, num_neighbors) 
% 
% Default arguments:
% DATA_ROOT = '~/data'
% save_suffix = '';

% Default arguments:
DATA_ROOT = '~/data';
save_suffix = '';

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'save_suffix'};
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
fprintf('class_indices: '); fprintf('%d ', class_indices); fprintf('\n');
fprintf('neighbor_name: %s\n', neighbor_name);
fprintf('num_neighbors: %d\n', num_neighbors);
fprintf('Default arguments:\n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('save_suffix: %s\n', save_suffix);

%----------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);


for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']), 'bboxes');
  n = numel(imdb.bboxes);
  
  neighbor_path = fullfile(root, neighbor_name, clname, num2str(num_neighbors));
  files = dir(fullfile(neighbor_path, '*_*.mat'));
  files = sort_chunk_name({files.name});

  markers = zeros(1,n);
  e = cell(n,1);
  for fidx = 1:numel(files)
    end_points = cellfun(@str2num, strsplit(files{fidx}(1:end-4), '_'));
    e(end_points(1):end_points(2)) = getfield(load(fullfile(neighbor_path, files{fidx})), 'e');
    assert(all(markers(end_points(1):end_points(2)) == 0));
    markers(end_points(1):end_points(2)) = 1;
  end
  assert(all(markers == 1));
  save_data_par(fullfile(root, neighbor_name, clname, [num2str(num_neighbors) save_suffix '.mat']), e, 'e');
end


end