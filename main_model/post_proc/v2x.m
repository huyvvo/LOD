function [x] = v2x(imgset, class_indices, solution_name, varargin)
% Get object indices from region rankings, with or without
% group membership condition and NMS.
%
% [perf] = v2x(imgset, class_indices, solution_name, varargin)
%
% Default arguments:
%
%   DATA_ROOT = '~/data';
%   nms_IoU = 0.3; % no NMS if nms_IoU >= 1
%   group_suppression = true;
%   imdb = [];
%

DATA_ROOT = '~/data';
nms_IoU = 0.3; % no NMS if nms_IoU >= 1
group_suppression = true;
imdb = [];

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'nms_IoU', 'group_suppression', 'imdb'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for name = validnames
  if ~isempty(strmatch(name{:},varnames))
    evalc(sprintf('%s=varvals{strmatch(name{:},varnames)}',name{:}));
  end
end

%--------------------------------------------------------------------

root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing class %s\n', clname);
  fprintf('Solution: %s\n', solution_name);
  
  %----------------------

  fprintf('Loading imdb ...');
  imdb_path = fullfile(root, clname, [clname, '_lite.mat']);
  if isempty(imdb)
    load_imdb_tic = tic;
    if group_suppression
      imdb = load(imdb_path, 'proposals', 'bboxes', 'root_feat_types_code', 'root_ids');
    else 
      imdb = load(imdb_path, 'proposals', 'bboxes');
    end
    fprintf('takes %.2f secs\n', toc(load_imdb_tic));
  else 
    fprintf('imdb found!\n');
  end

  n = numel(imdb.bboxes);
  num_props = cellfun(@(el) size(el,1), imdb.proposals);
  num_bboxes = cellfun(@(el) size(el,1), imdb.bboxes);
  if group_suppression
    group_code = get_group_code(imdb, n);
  end
  max_num_regions = max(5,max(num_bboxes));

  %----------------------

  sol_file = fullfile(root, clname, 'lsuod_results', solution_name);
  fprintf('Loading solutions ...');
  load_sol_tic = tic;
  v = load(sol_file);
  v = abs(v.v);
  v = mat2cell(v(:), num_props,1);
  fprintf('takes %.2f secs\n', toc(load_sol_tic));

  %----------------------

  fprintf('Computing x: ...\n');
  x_tic = tic;
  if isempty(gcp('nocreate'))
    if group_suppression
      x = v2x_group(v, imdb.proposals, group_code, ones(n,1)*max_num_regions, nms_IoU);
    else
      x = v2x_nogroup(v, imdb.proposals, ones(n,1)*max_num_regions, nms_IoU);
    end
  else 
    if group_suppression
      x = v2x_group_par(v, imdb.proposals, group_code, ones(n,1)*max_num_regions, nms_IoU);
    else 
      x = v2x_nogroup_par(v, imdb.proposals, ones(n,1)*max_num_regions, nms_IoU);
    end
  end
  fprintf('v2x in %.2f secs\n', toc(x_tic));
  
  if group_suppression
    save_path = fullfile(root, clname, 'lsuod_results', ...
                         strrep(solution_name, '/v_', sprintf('/x_group_nms_%.2f_', nms_IoU)));
  else
    save_path = fullfile(root, clname, 'lsuod_results', ...
                         strrep(solution_name, '/v_', sprintf('/x_nms_%.2f_', nms_IoU)));
  end
  fprintf('Saving x to %s\n', save_path);
  data = struct;
  data.x = x;
  savefile(save_path, data);
  
  
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [group_code] = get_group_code(imdb, n)
  if ~isfield(imdb, 'group_code')
    group_code = cell(n,1);
    for i = 1:n
      group_code{i} = imdb.root_feat_types_code{i}*1e6 + imdb.root_ids{i};
      [~, group_code{i}] = ismember(group_code{i}, unique(group_code{i}));
    end
  else 
    group_code = imdb.group_code;
  end
end

function [x] = v2x_group(v, proposals, group_code, num_regions, IoU)
  n = numel(proposals);
  x = cell(n,1);
  
  num_regions = mat2cell(num_regions, ones(1,n), 1);
  for i = 1:n 
    x{i} = group_nms(v{i}, proposals{i}, group_code{i}, num_regions{i}, IoU);
  end
end

function [x] = v2x_group_par(v, proposals, group_code, num_regions, IoU)
  n = numel(proposals);
  x = cell(n,1);
  
  num_regions = mat2cell(num_regions, ones(1,n), 1);
  parfor i = 1:n 
    x{i} = group_nms(v{i}, proposals{i}, group_code{i}, num_regions{i}, IoU);
  end
end

function [x] = v2x_nogroup(v, proposals, num_regions, IoU)
  n = numel(proposals);
  x = cell(n,1);
  
  num_regions = mat2cell(num_regions, ones(1,n), 1);
  for i = 1:n 
    [~,ids] = sort(v{i}, 'descend');
    x{i} = nms_single(ids, proposals{i}, IoU, num_regions{i});
  end
end

function [x] = v2x_nogroup_par(v, proposals, num_regions, IoU)
  n = numel(proposals);
  x = cell(n,1);
  
  num_regions = mat2cell(num_regions, ones(1,n), 1);
  parfor i = 1:n 
    [~,ids] = sort(v{i}, 'descend');
    x{i} = nms_single(ids, proposals{i}, IoU, num_regions{i});
  end
end
