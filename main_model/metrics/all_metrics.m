function [CorLoc, detRate, mABO, num_pos_regions, num_detected_objects, ABO] = all_metrics(proposals, bboxes, x, IoU_thresh)
% Compute all performance metrics.
% 
% [CorLoc, detRate, mABO, num_pos_regions, num_detected_objects, ABO] = all_metrics(proposals, bboxes, x, IoU_thresh)
%
% Parameters:
%
%   proposals: (n x 1) cell, proposals{i} contains proposals of image i.
%
%   bboxes: (n x 1) cell, bboxes{i} contains ground-truth bounding boxes of image i.
%
%   x: (n x 1) cell, x{i} contains the indices of the returned regions.
%
%   IoU_thresh: float, IoU threshold.
%
% Returns:
%
%   CorLoc: correct localization percentage.
%   detRate: detection rate.
%   mABO: mean Average Best Overlap.
%

n = numel(proposals);
num_pos_regions = cell(n,1);
num_detected_objects = cell(n,1);
ABO = cell(n,1);

if isempty(gcp('nocreate'))
  for i = 1:n 
    if numel(proposals{i}) > 0 && numel(x{i}) > 0 && numel(proposals{i}(x{i},:) > 0)
      IoU = pairwise_bbox_iou(proposals{i}(x{i},:), bboxes{i});
      num_pos_regions{i} = sum(max(IoU,[],2) >= IoU_thresh);
      num_detected_objects{i} = sum(max(IoU,[],1) >= IoU_thresh);
      ABO{i} = mean(max(IoU,[],1));
    else
      num_pos_regions{i} = 0;
      num_detected_objects{i} = 0; 
      ABO{i} = 0;
    end
  end
else
  parfor i = 1:n 
    if numel(proposals{i}) > 0 && numel(x{i}) > 0 && numel(proposals{i}(x{i},:) > 0)
      IoU = pairwise_bbox_iou(proposals{i}(x{i},:), bboxes{i});
      num_pos_regions{i} = sum(max(IoU,[],2) >= IoU_thresh);
      num_detected_objects{i} = sum(max(IoU,[],1) >= IoU_thresh);
      ABO{i} = mean(max(IoU,[],1));
    else
      num_pos_regions{i} = 0;
      num_detected_objects{i} = 0; 
      ABO{i} = 0;
    end
  end
end

num_pos_regions = cell2mat(num_pos_regions);
num_detected_objects = cell2mat(num_detected_objects);
ABO = cell2mat(ABO);
CorLoc = mean(num_pos_regions > 0);
detRate = sum(num_detected_objects) / sum(cellfun(@numel, bboxes)/4);
mABO = mean(ABO);

end
