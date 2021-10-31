function [new_imdb] = subsample_proposals(imdb, max_num_proposals)
%
% [new_imdb] = subsample_proposals(imdb, max_num_proposals)
%
% Limit the number of proposals for each image. This function use 
% should be limited to proposal sets generated from CNN features.
%
% Parameters:
%
%   imdb: struct, containing the following fields: 
%         'bboxes', 'images_size', 'proposals', 'root_ids', 
%         'root_feat_types_code'.
%
%   max_num_proposals: int, maximum number of proposals.
%

n = numel(imdb.proposals);
new_imdb = struct;
new_imdb.bboxes = imdb.bboxes;
new_imdb.images_size = imdb.images_size;
new_imdb.proposals = cell(n,1);
new_imdb.root_ids = cell(n,1);
new_imdb.root_feat_types_code = cell(n,1);

num_props = cellfun(@(el) size(el,1), imdb.proposals);
for i = 1:n 
  if num_props(i) <= max_num_proposals
    new_imdb.proposals{i} = imdb.proposals{i};
    new_imdb.root_ids{i} = imdb.root_ids{i};
    new_imdb.root_feat_types_code{i} = imdb.root_feat_types_code{i};
    continue;
  end
  ids = [1 sort(randperm(num_props(i)-1, max_num_proposals-1)+1)];
  new_imdb.proposals{i} = imdb.proposals{i}(ids,:);
  new_imdb.root_ids{i} = imdb.root_ids{i}(ids);
  new_imdb.root_feat_types_code{i} = imdb.root_feat_types_code{i}(ids);
end
end
