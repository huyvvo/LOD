function [ bboxes, num_pixels ] = get_boxes_from_segmentation(img, conn)
% GET_BOXES_FROM_SEGMENTATION
% [bboxes, num_pixels] = get_boxes_from_segmentation(img, conn)
%
% Get the tuples (x_min, y_min, x_max, y_max) representing the bounding box(es) in
% the image 'img'. Firstly, the algorithm finds connected component of foreground pixels
% then a bounding box is created for each connected component.
%
% Parameters:
%
%   img: matrix, an binary image whose non zero pixels correspond to object regions in the image.
%
%   conn: connectivity parameter of the MATLAB built-in function bwconncomp.
%
% Returns:
%
%   bboxes: array, a 4-tuples (x_min, y_min, x_max, y_max) representing the bounding boxes.
%

bboxes = [];
num_pixels = [];
components = bwconncomp(img, conn);
for compo = 1:numel(components.PixelIdxList)
  [Y, X] = ind2sub(size(img), components.PixelIdxList{compo});
  bboxes = [bboxes ; min(X) min(Y) max(X) max(Y)];
  num_pixels = [num_pixels ; numel(X)];
end


if numel(bboxes) == 0
  bboxes = [[0,0,0,0]];
  num_pixels = 0;
end


end