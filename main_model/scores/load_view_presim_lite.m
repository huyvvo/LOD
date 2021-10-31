function [viewInfo] = load_view_lite(img_size, seg, varargin)
% LOAD_VIEW_LITE
%
% [viewInfo] = load_view_lite(img_size, seg, feat, varargin)
% load image and make view info (proposals, descriptions, ...)
%
% Parameters:
%
%   img_size: pairs or 3-tuple of integers, size of the image.
%
%   seg: struct containing a field named 'coords' which are 
%       coordinates of proposals. 'coords' is an array of size
%       (N x 4) where N is the number of proposals in the image.
%
%
%   varargin: variable indicating which proposals to choose. Should be
%       set to "'conf', conf" in this project.
%
% Returns:
%
%   viewInfo: struct containing the following fields
%
%       viewInfo.img_size: 3 dimensional array.
%       viewInfo.frame: proposals represented as frames, see function
%           box2frame.
%       viewInfo.type:
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

viewInfo.img_size = img_size;
boxes = seg.coords;

viewInfo.frame = box2frame(boxes');
viewInfo.type = ones(1,size(viewInfo.frame,2),'int32');
viewInfo.patch = cell(0);
    
viewInfo.bbox = [ 1, 1, viewInfo.img_size(2), viewInfo.img_size(1) ]';
