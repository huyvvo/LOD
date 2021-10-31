function [proposals] = eliminate_border_boxes(proposals, images)
% ELIMINATE_BORDER_BOXES
%
% [proposals] = eliminate_border_boxes(proposals, images)
%
% Parameters:
%
%   proposals: (n x 1) cell, each contains proposals for one image.
%
%   images: (n x 1) cell, each contains an image.
%
% Returns:
%
% (n x 1) cell, each contains new proposals for an image.


assert(size(proposals, 1) == size(images, 1));
n = size(proposals, 1);
for i = 1:n
  h = size(images{i}, 1); w = size(images{i}, 2);
  valid_idx = proposals{i}(:,1) > 0.01*w & ...
              proposals{i}(:,3) < 0.99*w & ...
              proposals{i}(:,2) > 0.01*h & ...
              proposals{i}(:,4) < 0.99*h;
  valid_idx = valid_idx | (proposals{i}(:,1) < 0.01*w & ...
                           proposals{i}(:,3) > 0.99*w & ...
                           proposals{i}(:,2) < 0.01*h & ...
                           proposals{i}(:,4) > 0.99*h);
  proposals{i} = proposals{i}(valid_idx, :);
end

end