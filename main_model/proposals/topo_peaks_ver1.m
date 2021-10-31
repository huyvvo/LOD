function [peaks, max_peak_ids] = topo_peaks(heatmap, num_peaks, ws, peak_threshold)
% TOPO_PEAKS
%
% [peaks] = topo_peaks(heatmap, num_peaks)
%
% Find peaks in heatmap using persistence.
%
% Parameters:
%
%   heatmap: 2d array
%
%   num_peaks: int, number of peaks to take
%
%   ws: odd int, window size, size of the window around peaks used in
%       non-maximum suppression.
%
%   peak_threshold: float, only points whose value >= peak_threshold
%                 are considered as candidate for peaks.
%
% Returns:
%
%   peaks: (num_peaks x 1) array containing peak ids after non-maximum 
%          suppression.
%
%   max_peak_ids: column array containing ids of peaks before non-maximum
%                 suppression.
%

% INITIALIZATION
assert(mod(ws,2) == 1);
peaks = [];
% get size of heatmap
w = size(heatmap, 2); h = size(heatmap, 1);
% get number of points that will be considered
num_points = sum(heatmap(:) > 0);
% list of parents of nodes in the graph
parents = zeros(w*h, 1);
% list of birth and death time of points
living_time = zeros(w*h, 2);

% MAIN LOOP
% loop through points to get living time
[~, max_idx] = sort(heatmap(:), 'descend');
for time = 1:num_points
  % current point
  idx = max_idx(time); 

  % get neighbots indices
  neighbors = find_neighbors(idx, w, h); 
  % eliminate neighbors that have not been visited
  neighbors = neighbors(parents(neighbors) > 0);

  if isempty(neighbors) % isolated point
    parents(idx) = idx;
    living_time(idx,1) = time;
  else 
    % FIND
    % get roots of trees containing neighbors and intermidiate nodes
    % in-between
    [rts, visited_nodes] = Find(neighbors, parents); 
    % make sure that all nodes in 'rts' are roots
    assert(isequal(parents(rts), rts));
    % make sure that living_time of nodes in 'rts' are correctly set
    assert(all(living_time(rts,1) > 0));
    assert(all(living_time(rts,2) == 0));
    % find the oldest root
    [~, max_root_idx] = min(living_time(rts,1));
    max_root = rts(max_root_idx);

    % UNION
    % merge trees
    parents(visited_nodes) = max_root;
    living_time(rts, 2) = time;
    living_time(max_root, 2) = 0;

    parents(idx) = max_root;
    living_time(idx,:) = [time, time];
  end
end

% NON-MAXIMUM SUPPRESION
living_time(living_time(:,1)>0 & living_time(:,2)==0, 2) = num_points;
age = living_time(:,2) - living_time(:,1);
age(heatmap(:)==0,:) = -1;
[~, max_peak_ids] = sort(age, 'descend');

mask = zeros(size(heatmap));
for idx = max_peak_ids(1:num_points)'
  if age(idx) <= 0
    break;
  end
  if heatmap(idx) < peak_threshold
    continue;
  end
  ws_ids = get_square_indices(idx, w, h, ws);
  if any(mask(ws_ids) == 1)
    continue;
  else 
    mask(ws_ids) = 1;
    peaks = [peaks idx];
  end
  if numel(peaks) == num_peaks
    break;
  end
end
end


%----------------------------------------------
function [rts, visited_nodes] = Find(ids, parents)
% find roots of the trees containing ids
% all nodes in 'ids' must be visited previously
assert(all(parents(ids) > 0));
rts = [];
visited_nodes = [];
for id = ids
  cn = id; % current node
  visited_nodes = [visited_nodes cn];
  while cn ~= parents(cn)
    cn = parents(cn);
    visited_nodes = [visited_nodes cn];
  end
  rts = [rts; cn];
end
visited_nodes = unique(visited_nodes);
end

%----------------------------------------------
function [neighbors] = find_neighbors(idx, w, h)
% find indices of neighbors the cell 'idx' in the image of size
% (h x w)
[i,j] = ind2sub([h,w],idx);
I = [i-1 i-1 i-1 i i i+1 i+1 i+1];
J = [j-1 j j+1 j-1 j+1 j-1 j j+1];
valid = (I>0)&(I<=h)&(J>0)&(J<=w);
neighbors = sub2ind([h,w],I(valid),J(valid));
end

%----------------------------------------------
function [ids] = get_square_indices(idx, w, h, ws)
% get indices of points in the square centered at 'idx'
% and has size (ws x ws).
[i,j] = ind2sub([h,w], idx);

I = repelem([i-floor(ws/2):i+floor(ws/2)]', ws, 1);
J = repmat([j-floor(ws/2):j+floor(ws/2)]', ws, 1);

valid_idx = (I>0) & (I<=h) & (J>0) & (J<=w);
I = I(valid_idx);
J = J(valid_idx);

ids = sub2ind([h,w], I, J);
end

