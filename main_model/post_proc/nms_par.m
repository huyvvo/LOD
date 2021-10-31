function [x_opt] = nms_par(x, proposals, iou_thresh, k)
% NMS_PAR
%
% Perform non-maximum suppression in parallel on x such that there are at most
% k regions retained in each image.
%
% Parameters:
%
%   x: (n x 1) cell, x{i} contains indices of regions in image i
%      in an order depending on a certain measure of goodness.
%
%   proposals: (n x 1) cell, cell proposals{i} contains regions 
%              of image i.
%
%   iou_thresh: float, a threshold to decide if a region is 
%               suppressed. A region is suppressed if its IoU with
%               one of the regions previously chosen is greater than
%               'iou_thresh'
%
%   k: int, the maximum number of regions to retain in each image
%      after the NMS.
%
% Returns:
%
%   x_opt: (n x 1) cell, cell x_opt{i} contains indices of regions 
%          retained in image i after the NMS.

n = size(x, 1);
assert(size(proposals, 1) == n);
x_opt = cell(size(x));
parfor i = 1:n 
  x_opt{i} = nms_single(x{i}, proposals{i}, iou_thresh, k);
end