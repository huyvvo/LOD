function [] = gather_norm_factor(norm_factor_path)
%
% [] = gather_norm_factor(norm_factor_path)
% 
  if exist(fullfile(norm_factor_path, 'norm_factor.mat'), 'file') == 2
    fprintf('norm_factor.mat exists in %s, recomputing it!\n', norm_factor_path);
    system(sprintf('rm %s', fullfile(norm_factor_path, 'norm_factor.mat')));
  end
  files = dir(fullfile(norm_factor_path, '*.mat'));
  files = sort_chunk_name({files.name});
  n = numel(files);
  fprintf('There are %d parts\n', n);
  norm_factor = cell(n,1);
  if isempty(gcp('nocreate'))
    for fidx = 1:n 
      fprintf('%d ', fidx);
      norm_factor{fidx} = getfield(load(fullfile(norm_factor_path, files{fidx})), 'norm_factor');
    end
  else
    parfor fidx = 1:n 
      fprintf('%d ', fidx);
      norm_factor{fidx} = getfield(load(fullfile(norm_factor_path, files{fidx})), 'norm_factor');
    end
  end
  norm_factor = full(cell2mat(norm_factor));
  fprintf('Saving to %s\n', fullfile(norm_factor_path, 'norm_factor.mat'));
  save_data_par(fullfile(norm_factor_path, 'norm_factor.mat'), norm_factor, 'norm_factor');
end