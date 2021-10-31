function [S, sum_row_S] = gather_matrix_parts_SSMC_symmetric(imgset, num_parts, class_indices, part_indices, score_name, varargin)
%
% [S, sum_row_S] = gather_matrix_parts_SSMC_symmetric(imgset, num_parts, class_indices, part_indices, score_name, varargin)
%
% Default parameters:
%
% DATA_ROOT = '~/data';
% num_keep = Inf;
% num_keep_text = '1000';
% score_type = 'confidence_symmetric';
% save_dir = '';
% save_result = true;
%

DATA_ROOT = '~/data';
num_keep = Inf;
num_keep_text = '1000';
score_type = 'confidence_symmetric';
save_dir = '';
save_result = true;

varnames = varargin(1:2:length(varargin));
varvals = varargin(2:2:length(varargin));
validnames = {'DATA_ROOT', 'num_keep', 'num_keep_text', 'score_type', 'save_dir', 'save_result'};
for name = varnames
  if ~any(cellfun(@(el) strcmp(name{:}, el), validnames))
    error(sprintf('"%s" is not a valid argument!', name{:}));
  end
end
for varidx = 1:numel(varnames)
  evalc(sprintf('%s=varvals{%d}',varnames{varidx}, varidx));
end

if isempty(save_dir)
  save_dir = [score_type, '_imdb_matrix_SSMC'];
end

fprintf('Arguments:\n');
fprintf('imgset: %s\n', imgset);
fprintf('num parts: %d\n', num_parts);
fprintf('class_indices: '); fprintf('%d ', class_indices); fprintf('\n');
fprintf('part indices: '); fprintf('%d ', part_indices); fprintf('\n');
fprintf('score_name: %s\n', score_name);
fprintf('Default arguments: \n');
fprintf('DATA_ROOT: %s\n', DATA_ROOT);
fprintf('num_keep: %s\n', string(num_keep));
fprintf('num_keep_text: %s\n', num_keep_text);
fprintf('score_type: %s\n', score_type);
fprintf('save_dir: %s\n', save_dir);
fprintf('save_result: %s\n', string(save_result));

%--------------------------------------------------------------------------


root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

for cl = class_indices
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);

  num_props = getfield(load(fullfile(root, clname, [clname, '_lite.mat']), 'num_props'), 'num_props');
  n = numel(num_props);
  %%%%%%%%%%%%
  info = whos;
  fprintf('Memory: %.2f Gb\n', sum(cell2mat({info.bytes}))/1024^3);
  %%%%%%%%%%%%
  pivots = [0 cumsum(num_props')];
  N = sum(num_props);

  [image_edges, prop_edges] = get_part_info(num_props, num_parts);

  indices = getfield(load(fullfile(root, clname, score_type, score_name, 'indices.mat')), 'indices');

  score_path = fullfile(root, clname, score_type, score_name);
  files = dir(fullfile(score_path, '*_*.mat')); 
  files = sort_chunk_name({files.name});
  num_files = numel(files);

  pos = strfind(score_name, '_');
  pos = pos(end);
  score_name_to_save = sprintf('%s_%s', score_name(1:pos-1), num_keep_text);
  score_path_to_save = fullfile(root, clname, save_dir, score_name_to_save);
  norm_factor_path_to_save = fullfile(root, clname, [save_dir, '_norm_factor'], score_name_to_save);
  
  mkdir(score_path_to_save);
  mkdir(norm_factor_path_to_save);

  for part_idx = part_indices
    part_tic = tic;
    img_first = image_edges{part_idx}(1);
    img_last = image_edges{part_idx}(2);
    row_first = min(find(indices(:,1)==img_first));
    row_last = max(find(indices(:,1)==img_last));

    SI = {}; SJ = {}; SV = {};
    for i = 1:num_files
      % read indices from filenames
      filename = files{i};
      end_points = cellfun(@str2num, strsplit(filename(1:end-4), '_'));
      if end_points(2) < row_first || end_points(1) > row_last
        continue;
      end
      fprintf('Part: %d, processing for rows %d to %d\n', part_idx, end_points(1), end_points(2));
      scores = getfield(load(fullfile(score_path, filename)), 'data');
      for row = [max(end_points(1),row_first):min(end_points(2),row_last)]
        r = indices(row,1); c = indices(row,2);
        assert(r >= img_first && r <= img_last);
        if num_keep ~= Inf
          current_scores = sparsify_matrix(scores{row-end_points(1)+1}, num_keep);
        else
          current_scores = scores{row-end_points(1)+1};
        end
        [I,J,V] = find(current_scores);
        clear current_scores;
        I = I + pivots(r);
        J = J + pivots(c);
        SI{end+1,1} = I(:);
        SJ{end+1,1} = J(:);
        SV{end+1,1} = V(:);
        scores{row-end_points(1)+1} = [];
      end
    end
    fprintf('Finish part %d in %.2f secs\n', part_idx, toc(part_tic));  
    %%%%%%%%%%%%
    info = whos;
    fprintf('Memory: %.2f Gb\n', sum(cell2mat({info.bytes}))/1024^3);
    %%%%%%%%%%%%

    SI = round(cell2mat(SI));
    SJ = round(cell2mat(SJ));
    SV = cell2mat(SV);
    first_index = prop_edges{part_idx}(1);
    second_index = prop_edges{part_idx}(2);
    assert(first_index <= min(SI) && max(SI) <= second_index);

    S = SparseSquareMatrixChunk(SI,SJ,SV,N,true,true,first_index,second_index);
    clear SI SJ SV;
    
    sum_row_S = SSMC_sum_by_row(S);
    if save_result
      fprintf('Saving to %s\n', fullfile(score_path_to_save, sprintf('%d_%d.mat', img_first, img_last)));
      save_data_par(fullfile(score_path_to_save, sprintf('%d_%d.mat', img_first, img_last)), S, 'S');
      save_data_par(fullfile(norm_factor_path_to_save, sprintf('%d_%d.mat', img_first, img_last)), sum_row_S, 'norm_factor');
    end

    clear S;
  end
end

end