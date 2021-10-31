function [processed_feat] = process_deepfeat(feat, similarity_measure)
% PROCESS_DEEPFEAT
%
% [processed_feat] = process_deepfeat(feat, similarity_measure)
%
% Parameters:
%
%   feat: 4 dimensional tensor containing deep features for proposals
%         in an image. The feature of each proposal is a 3-dimensional
%         tensor of size (num_channels, heigh, width).
%
%   similarity_measure: how to process the features.
%
% Returns:
%
%   2 dimentional array whose rows are feature vectors of proposals.

assert(numel(size(feat)) == 4);
feat = permute(feat, [1,3,4,2]); 

if strcmp(similarity_measure, 'sp') == 1
  feat = reshape(feat, size(feat, 1), []);
  processed_feat = feat;
elseif strcmp(similarity_measure, 'cos') == 1
  feat = reshape(feat, size(feat, 1), []);
  processed_feat = feat./sqrt(sum(feat.*feat, 2));
elseif  strcmp(similarity_measure, 'spatial_cos')
  feat = reshape(feat, size(feat,1), [], size(feat,4));
  processed_feat = feat./sqrt(sum(feat.^2,3));
  processed_feat = reshape(processed_feat, size(processed_feat,1), []);
elseif strcmp(similarity_measure, '01') == 1
  feat = reshape(feat, size(feat, 1), []);
  processed_feat = feat./max(feat(:));
elseif strcmp(similarity_measure, 'spatial_01')
  feat = reshape(feat, size(feat,1), [], size(feat,4));
  processed_feat = feat./max(feat,3);
  processed_feat = reshape(processed_feat, size(processed_feat,1), []);
elseif strcmp(similarity_measure, 'sqrt') == 1
  feat = reshape(feat, size(feat, 1), []);
  feat = sqrt(feat);
  processed_feat = feat./max(feat(:));
elseif strcmp(similarity_measure, 'log')
  feat = reshape(feat, size(feat, 1), []);
  feat = log(feat + 1);
  processed_feat = feat./max(feat(:));
else 
  error('Unknown similarity_measure');
end
