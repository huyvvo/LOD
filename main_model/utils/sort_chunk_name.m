function [res] = sort_chunk_name(chunk_names)
%
%
% Parameters:
%
%   chunk_names: cell, elements are string of types a_b.mat
%
% Returns:
%
% cell, chunk names in increasing lexicographical order of pairs (a,b)

chunk_names_col = chunk_names(:);
chunk_code = cellfun(@(el) cellfun(@str2num, strsplit(el(1:end-4), '_')), ...
                     chunk_names_col, 'Uni', false);
chunk_code = cell2mat(chunk_code);
[~, ids] = sortrows(chunk_code);
res = chunk_names(ids);

end