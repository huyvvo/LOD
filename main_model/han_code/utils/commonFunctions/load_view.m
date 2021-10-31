function [ viewInfo ] = load_view( img, seg, feat, varargin )
% [ viewInfo ] = load_view( img, seg, feat, varargin )
% load image and make view info (proposals, descriptions, ...)
%
% Parameters:
%
%   img: image, 3 dimensional array.
%
%   seg: struct containing a field named 'coords' which are 
%       coordinates of proposals. 'coords' is an array of size
%       (N x 4) where N is the number of proposals in the image.
%
%   feat: struct containing a field named 'hist' which is an array 
%       of size (N x feat_dim) where N is the number of 
%       proposals and feat_dim is the length of proposals' features.
%       Row i of 'feat' is the feature corresponding to proposal i.
%
%   varargin: variable indicating which proposals to choose. Should be
%       set to "'conf', conf" in this project.
%
% Returns:
%
%   viewInfo: struct containing the following fields
%
%       viewInfo.img: 3 dimensional array.
%       viewInfo.idx2ori: array, index of chosen proposals.
%       viewInfo.frame: proposals represented as frames, see function
%           box2frame.
%       viewInfo.type:
%       viewInfo.desc: HOG description of chosen proposals.
%       viewInfo.bbox: the rectangle represents the whole image.

% ----------------------------------------------------------------------
% READ PARAEMTERS

conf = [];
cand = [];
for k=1:2:length(varargin)
  opt=lower(varargin{k}) ;
  arg=varargin{k+1} ;
  switch opt
    case 'conf'
      conf = arg;
    case 'cand'
      cand = arg;  
    otherwise
      error(sprintf('Unknown option ''%s''', opt)) ;
  end
end

viewInfo.img = img;
boxes = seg.coords;

if size(viewInfo.img,3) == 3
    viewInfo.img_gray = rgb2gray(viewInfo.img);
elseif size(featInfo.img,3) == 1
    viewInfo.img_gray = viewInfo.img;
    viewInfo.img = repmat( viewInfo.img, [ 1 1 3 ]);
else
    err([ 'wrong image file!: ' filePathName ]);
end

% -----------------------------------------------------------------------------
% CHOOSE PROPOSALS TO RETAIN

% if isempty(cand)
%     bValid1 = boxes(:,1) > size(viewInfo.img,2) * 0.01 & ...
%               boxes(:,3) < size(viewInfo.img,2) * 0.99 & ...
%               boxes(:,2) > size(viewInfo.img,1) * 0.01 & ...
%               boxes(:,4) < size(viewInfo.img,1) * 0.99;
%     bValid2 = boxes(:,1) < size(viewInfo.img,2) * 0.01 & ...
%               boxes(:,3) > size(viewInfo.img,2) * 0.99 & ...
%               boxes(:,2) < size(viewInfo.img,1) * 0.01 & ...
%               boxes(:,4) > size(viewInfo.img,1) * 0.99;
%     idxValid = find(bValid1 | bValid2);
% else
%     idxValid = cand;
% end

idxValid = 1:size(boxes, 1); % choose all proposals
boxes = boxes(idxValid,:);

viewInfo.idx2ori = int32(idxValid);
viewInfo.frame = box2frame(boxes');
viewInfo.type = ones(1,size(viewInfo.frame,2),'int32');
viewInfo.desc = single(feat.hist(idxValid,:)');
viewInfo.patch = cell(0);

if isfield(feat, 'mask')
    viewInfo.mask = feat.mask(idxValid)';
end
if isfield(feat, 'hist_mask')
    viewInfo.desc_mask = single(feat.hist_mask(idxValid,:)');
end
    
viewInfo.bbox = [ 1, 1, size(viewInfo.img,2), size(viewInfo.img,1) ]';
