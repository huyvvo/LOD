function [ corloc, iou_score ] = CorLoc(proposals, bboxes, x, threshold)
% CORLOC
% [ corloc, iou_score ] = CorLoc(proposals, bbox, x, thresh)
%
% Compute CorLoc score.
%
% Parameters:
%
%   proposals: (n x 1) cell, proposals{i} contains the proposals 
%              of image i. Proposals are in the format 
%              [xmin, ymin, xmax, ymax] where (xmin, ymin) is the 
%              coordinate of the top left corner, (xmax, ymax) is
%              the coordinate of the bottom right corner.
%
%   bboxes: (n x 1) cell, bboxes{i} contains ground truth bboxes
%           of image i. Bboxes are in the same format as proposals.
%
%   x:  (n x 1) cell, x{i} contains indices of regions retained in
%       in image i.
%
%   threshold: double, threshold for IOU score.
%
%
% Returns:
%
%   corloc: double, CorLoc score.
%
%   iou_score: (n x 1) cell, iou{i} is an array of size 
%              (num_bboxes x num_proposals), containing the IoU between 
%              retained proposals in image i and its bboxes. 
%

n = size(x, 1);
iou_score = cell(n,1);
for i = 1:n
  positive_boxes = proposals{i}(x{i},:);
  iou = [];
  for box_idx = 1:size(bboxes{i},1)
    iou = [iou, bbox_iou(positive_boxes, bboxes{i}(box_idx,:))];
  end
  iou_score{i} = iou';
end
corloc = sum(cellfun(@(el) max(el(:)) >= threshold, iou_score))/n;
end

