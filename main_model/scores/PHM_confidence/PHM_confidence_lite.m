function [confidenceMap] = PHM_confidence_lite( PHM_func, ...
                            img_size_A, img_size_B, proposal_A, proposal_B, ...
                            feat_A, feat_B, feature_type, bg)
% PHM_CONFIDENCE_LITE
%
% [confidenceMap] = PHM_confidence_lite(PHM_func, 
%                            img_size_A, img_size_B, proposal_A, proposal_B, ...
%                            feat_A, feat_B, feature_type)
%
% Compute confidence score between regions in two images.
%
% Parameters:
%
%   PHM_func: handle to the function to compute PHM.
%
%   img_size_A: pair of 3-tuple of integers, size of image A.
%
%   img_size_B: pair of 3-tuple of integers, size of image B.
%
%   proposal_A: (kA x 4) matrix, coordinates of proposals in img_size_A.
%
%   proposal_B: (kB x 4) matrix, coordinates of proposals in img_size_B.
%
%   feat_A: (kA x d) matrix, features of regions of image A.
%
%   feat_B: (kB x d) matrix, features of regions of image B.
%
%   feature_type: string, specify how to process cnn feature.
%
% Returns:
%
% (kA x kB) matrix containing confidence scores of matches between regions.
%

global conf; % required by load_view, but not used

% create required structures for function extract_segfeat_hog
op_A.coords = proposal_A;
op_B.coords = proposal_B; 

struct_feat_A.hist = feat_A;
struct_feat_B.hist = feat_B;

% fprintf(' - %s matching... ', 'PHM');

% options for matching
opt.bDeleteByAspect = true;
opt.bDensityAware = false;
opt.bSimVote = true;
opt.bVoteExp = true;
opt.feature = feature_type;
if exist('bg', 'var') == 1
  opt.bg = bg;
else 
  opt.bg = [];
end


viewA = load_view_lite(img_size_A, op_A, struct_feat_A, 'conf', conf);
viewB = load_view_lite(img_size_B, op_B, struct_feat_B, 'conf', conf);

% compute confidence of pairwise matches between regions
confidenceMap = PHM_func( viewA, viewB, opt );

end
