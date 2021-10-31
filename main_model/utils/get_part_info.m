function [image_edges, prop_edges] = get_part_info(num_props, num_parts)
% Compute first and last indices if the images and proposals in each part
%
% [image_edges, prop_edges] = get_part_info(num_props, num_parts)
%
% Parameters:
%
%   num_props: (n x 1) array, number of proposals in the images.
%   num_parts: int, number of parts.
%   
  n = numel(num_props);

  part_sizes = ones(1, num_parts)*floor(n/num_parts);
  part_sizes(1:mod(n,num_parts)) = part_sizes(1:mod(n,num_parts))+1;
  
  p = [0, cumsum(part_sizes)];
  image_edges = arrayfun(@(i) [p(i)+1 p(i+1)], 1:num_parts, 'Uni', false);
  
  p = [0, cumsum(num_props')];
  prop_edges = arrayfun(@(i) [p(image_edges{i}(1))+1 p(image_edges{i}(2)+1)], 1:num_parts, 'Uni', false);

end