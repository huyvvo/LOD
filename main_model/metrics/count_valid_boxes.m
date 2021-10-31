function [ count, iou ] = count_valid_boxes(proposals, bboxes, x, threshold)
% COUNT_VALID_BOXES
% [ count, iou ] = count_valid_boxes(proposals, bbox, x, threshold)
%
% Count the number of proposals retained in 'x' that have the iou score compared to 
% ground truth boxes greater than or equal to 'threshold'.
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
%		x:  (n x 1) cell, x{i} contains indices of regions retained in
%       in image i.
%
%   threshold: double, threshold for IOU score.
%
% Returns:
%
%		count: (n x 1) array, the number of positive retained proposals 
%          in each image.
%
%   iou: (n x 1) cell, iou{i} is an array of size 
%        (num_bboxes x num_proposals), containing the IoU between 
%        retained proposals in image i and its bboxes. 
% 

n = size(x, 1);
count = cell(n,1);
iou_score = cell(n,1);
for i = 1:n
	positive_boxes = proposals{i}(x{i},:);
  iou_score{i} = pairwise_bbox_iou(positive_boxes, bboxes{i})';
  count{i} = sum(max(iou_score{i}, [], 1) >= threshold);
end
count = cell2mat(count);
end