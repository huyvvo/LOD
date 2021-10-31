function [ids_opt] = nms_single(ids, proposals, iou_thresh, k)
% [ids_opt] = nms_single(ids, proposals, iou_thresh, k)
% NMS
%
% Perform non-maximum suppression on sorted array 'ids' of regions
%  such that there are at most k regons retained in 'proposals'.
%
% Parameters:
%
%   ids: array, contains indices of regions in 'proposals'
%      in an order depending on a certain measure of goodness.
%
%   proposals: (N x 4) array. contains regions proposals.
%
%   iou_thresh: float, a threshold to decide if a region is 
%               suppressed. A region is suppressed if its IoU with
%               one of the regions previously chosen is greater than
%               'iou_thresh'
%
%   k: int, the maximum number of regions to retain after the NMS.
%
% Returns:
%
%   ids_opt: array, contains indices of regions retained after NMS.

if isempty(ids)
  ids_opt = [];
  return
end
ids_opt = [ids(1)];
for j = 2:numel(ids)
  if numel(ids_opt) == k
    break;
  end
  iou = bbox_iou(proposals(ids(j), :), proposals(ids_opt, :));
  if max(iou) < iou_thresh
    ids_opt(end+1) = ids(j);
  end
end 