function [proposals, root_ids] = get_proposals_from_heatmap(heatmap, imgsize, qlist, init_level, conn)
% GET_PROPOSALS_FROM_HEATMAP
%
% This function generates proposals from a local saliency map.
%
% [proposals] = get_proposals_from_heatmap(heatmap, imgsize, qlist, init_level, conn)
%
% Parameters:
%
%   heatmap: 2d array, a saliency maps.
%
%   imgsize: 2-tuple of integers showing the height and width of the image.
%
%   qlist: a list of thresholds on the local saliency map. Instead of using directly saliency 
%          values as thresholds, we use the numbers of pixels to retain on the heatmap to 
%          represent the thresholds.
%
%   init_value: A real number indicating the maximum saliency value that is considered as a
%               threshold on the saliency map.
%
%   conn: an integer used to define the connectivity between pixels in the saliency map.
%
% Returns:
%
%   proposals: (N x 4) array whose rows are coordinates of proposals.
%   
%   root_ids: the index of the most salient pixel in the cluster from which each proposal is
%          created.
%

% the max threshold that is considered in the heatmap
init_level = min(init_level, max(heatmap(:)));
% eliminate values of q in qlist that correspond to thresholds larger than init_elevel
init_level_map = heatmap >= init_level;
components = bwconncomp(init_level_map, conn);
init_q = numel(components.PixelIdxList);
qlist = qlist(qlist >= init_q);

% sort values in the heatmap in decreasing order
[~, max_ids] = sort(heatmap(:), 'descend');
proposals = [];
root_ids = [];
for q = qlist
  level_map = zeros(size(heatmap));
  level_map(max_ids(1:q)) = heatmap(max_ids(1:q));

  % eliminate clusters whose the max value is smaller than init_value
  % this is an extension to what described in the paper: Instead of keeping
  % only the cluster containing the local maximum, we keep also clusters 
  % whos the max value is greater than init_value
  components = bwconncomp(level_map > 0, conn);
  % array indicating if a connect component is valid
  valid_compo_ids = zeros(1, numel(components.PixelIdxList));
  % array containing the index of the most salient pixel in each connected component
  max_idx_compo = zeros(1, numel(components.PixelIdxList));
  for cpid = 1:numel(valid_compo_ids)
    pixel_ids = components.PixelIdxList{cpid};
    [max_saliency, max_saliency_index] = max(level_map(pixel_ids));
    max_idx_compo(cpid) = pixel_ids(max_saliency_index);
    if max_saliency >= init_level
      valid_compo_ids(cpid) = 1;
    end
  end
  % create a new saliency map with only valid clusters as described above
  new_level_map = zeros(size(level_map));
  for cpid = find(valid_compo_ids)
    new_level_map(components.PixelIdxList{cpid}) = level_map(components.PixelIdxList{cpid});
  end
  max_idx_compo = max_idx_compo(find(valid_compo_ids));
  % get proposals from a newly create saliency map
  new_level_map = new_level_map > 0;
  new_level_map = imresize(new_level_map, imgsize, 'nearest');
  [boxes, num_pixels] = get_boxes_from_segmentation(new_level_map, conn);
  proposals = [proposals ; boxes];  
  current_root_ids = map_boxes_to_root_ids(boxes, max_idx_compo, imgsize, size(heatmap));
  root_ids = [root_ids ; current_root_ids];
end


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [root_ids] = map_boxes_to_root_ids(boxes, root_ids_list, imgsize, heatmap_size)
  % rescale boxes
  rescaling_ratio = heatmap_size ./ imgsize; % (y_ratio, x_ratio)
  boxes = [floor(boxes(:,1)*rescaling_ratio(2)), floor(boxes(:,2)*rescaling_ratio(1)), ...
           ceil(boxes(:,3)*rescaling_ratio(2)), ceil(boxes(:,4)*rescaling_ratio(1))];
  % find coordinates [row, col] of roots
  [root_coor_row_list, root_coor_col_list] = ind2sub(heatmap_size, root_ids_list');
  root_coor_list = [root_coor_row_list, root_coor_col_list];
  % find root index for boxes
  root_ids = [];
  for i = 1:size(boxes,1)
    % find if roots are in the current box
    in_bbox = check_in_bbox(root_coor_list, boxes(i,:));
    % find a root that is in the current box and add to the root_ids list
    % if no root found, 0 is returned
    current_root_idx = find(in_bbox);
    if isempty(current_root_idx)
      root_ids = [root_ids; 0];
    else
      root_ids = [root_ids; root_ids_list(current_root_idx(1))]; 
    end
  end
  assert(isempty(setdiff(root_ids, [root_ids_list, 0])));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [in_bbox] = check_in_bbox(root_coor_list, bbox)
  % each root_coor is [row, col] or [y, x] of a root
  in_bbox =  root_coor_list(:,1) >= bbox(2) & root_coor_list(:,1) <= bbox(4) & ...
             root_coor_list(:,2) >= bbox(1) & root_coor_list(:,2) <= bbox(3);
end