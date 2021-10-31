function [] = compress_proposals_edgeboxes(imgset, proposal_name, class_indices, varargin)
%
% function [] = compress_proposals_edgeboxes(imgset, proposal_name, class_indices, varargin)
%
% Default parameters:
% DATA_ROOT = '~/data'

% Default parameters
DATA_ROOT = '~/data';

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT'};
for name = varnames
  if ~all(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for name = validnames
  if ~isempty(strmatch(name{:},varnames))
    evalc(sprintf('%s=varvals{strmatch(name{:},varnames)}',name{:}));
  end
end

fprintf('Arguments:\n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('imgset: %s\n', imgset);
fprintf('class_indices: %d ', class_indices); fprintf('\n');
fprintf('proposal_name: %s\n', proposal_name);

%---------------------------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Compress proposals for class %s\n', clname);
  proposal_path = fullfile(root, clname, [clname, '_', proposal_name]);
  files = dir(fullfile(proposal_path, '*_*.mat'));
  files = {files.name};
  num_files = numel(files);
  if num_files == 0
    error('Proposals do not exists in %s!\n', proposal_path);
  end

  imdb = struct;
  for i = 1:num_files
    filename = files{i};
    end_points = cellfun(@str2num, strsplit(filename(1:end-4), '_'));
    rows = end_points(1):end_points(2);
    fprintf('Processing for rows %d to %d\n', end_points);
    data = load(fullfile(proposal_path, filename));
    fields = fieldnames(data);
    for fidx = 1:numel(fields)
      imdb = setfield(imdb, fields{fidx}, {rows, 1}, getfield(data, fields{fidx}, {rows}));
    end
  end
  fprintf('\n');  
  save_path = fullfile(root, clname, [clname, '_lite.mat']);
  fprintf('Saving to %s\n', save_path);
  savefile(save_path, imdb);
end 