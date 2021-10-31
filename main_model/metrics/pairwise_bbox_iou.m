function [IoU] = pairwise_bbox_iou(bboxesA, bboxesB)
%
% [IoU] = pairwise_bbox_iou(bboxesA, bboxesB) 
%
% Parameters:
%
%   bboxesA: (nA x 4) array
%   bboxesB: (nB x 4) array
%
% Returns:
%
% IoU: (nA x nB) array
%
  bboxesA = double(bboxesA);
  bboxesB = double(bboxesB);
  
  nA = size(bboxesA,1);
  nB = size(bboxesB,1);

  bboxesA = repmat(bboxesA, nB, 1);
  bboxesB = repelem(bboxesB, nA, 1);
  areaA = (bboxesA(:,3)-bboxesA(:,1)+1) .* (bboxesA(:,4)-bboxesA(:,2)+1);
  areaB = (bboxesB(:,3)-bboxesB(:,1)+1) .* (bboxesB(:,4)-bboxesB(:,2)+1);

  bboxesi = [max(bboxesA(:,1), bboxesB(:,1)) ... 
             max(bboxesA(:,2), bboxesB(:,2)) ...
             min(bboxesA(:,3), bboxesB(:,3)) ...
             min(bboxesA(:,4), bboxesB(:,4)) ...
            ];
  areai = max(bboxesi(:,3)-bboxesi(:,1)+1,0) .* max(bboxesi(:,4)-bboxesi(:,2)+1,0);
  
  IoU = areai ./ (areaA + areaB - areai);
  IoU = reshape(IoU, [nA,nB]);
end