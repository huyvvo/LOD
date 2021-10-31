function [x] = group_nms(score, proposals, group_code, k, nms_IoU)
%
% [x] = group_nms(score, proposals, group_code, k, nms_IoU)
%
% Parameters:
%
%   score: (N,) array
%   proposals: (N x 4) array
%   group_code: (N,) array
%   k: int
%   nms_IoU: float

  [~, ids] = sort(score, 'descend');
  ids = ids(:)';
  x = [];
  for i = ids
    if isempty(x) || (~ismember(group_code(i), group_code(x)) && max(bbox_iou(proposals(i,:), proposals(x,:))) <= nms_IoU)
      x(end+1) = i;
      if numel(x) == k
        break;
      end
    end
  end
end
