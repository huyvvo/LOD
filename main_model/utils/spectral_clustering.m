function [clusters] = spectral_clustering(A, k)
% SPECTRAL_CLUSTERING
%
% [clusters] = spectral_clustering(A, k)
%
% Parameters:
%
% A: a symmetric matrix, the adjacency metrix of a undirected graph.
%
% k: int, number of clusters
%
% Returns:
%
% clusters: (k x 1) cell, each cell containing the indices of nodes
%       in the corresponding cluster.

A = A/max(A(:));
A = exp(A);
D = sum(A);
L_unn = diag(D) - A;
L_norm = diag(D.^(-0.5)) * L_unn * diag(D.^(-0.5));
[VE, VA] = eigs(L_norm);
[~, max_idx] = sort(VA, 'descend');
VE = VE(:, max_idx(1:k));
cluster_assignment = kmeans(VE, k);
clusters = cell(k,1);
for i = 1:size(A, 1)
  clusters{cluster_assignment(i)} = [clusters{cluster_assignment(i)}, ...
                                     i];
end
