function [nr] = compute_num_rows(e_path)
% Compute number of image pairs to compute PHM score from 
% the lists if neighbors of the images.
%
% [nr] = compute_num_rows(e_path)
%
% Parameters:
%
%  e_path: string, path to neighbor .mat file
%
  e = getfield(load(e_path), 'e');
  n = numel(e);
  actual_num_nb = cellfun(@numel, e);
  indices = [repelem([1:n]', actual_num_nb) reshape(cell2mat(e'), [], 1)];
  indices = [min(indices'); max(indices')]';
  indices = unique(indices, 'rows');
  nr = size(indices,1);
end