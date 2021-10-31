function [sparse_S] = sparsify_matrix(S, num_keep)
% SPARSIFY_MATRIX
% 
% [sparse_S] = sparsify_matrix(S, num_keep)
%
% Get a new sparse matrix with only 'num_keep' top elements from 'S'.
%
% Paramters:
%
%   S: a matrix.
%
%   num_keep: int, number of top elements to be kept.
%
% Returns:
%
% A 'sparse matrix' containing only top elements in 'S'.

[~,max_idx] = sort(S(:), 'descend');
max_idx = max_idx(1:min(num_keep, end));
[I,J] = ind2sub(size(S), max_idx);
sparse_S = sparse(I,J,double(S(max_idx)),size(S,1),size(S,2));
end