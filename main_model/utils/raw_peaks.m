function [peaks] = raw_peaks(heatmap, num_peaks, ws, peak_threshold)
% RAW_PEAKS
%
% [peaks] = raw_peaks(heatmap, num_peaks, ws, peak_threshold)
%
% Find peaks in heatmap using saliency.
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
%                   are considered as candidate for peaks.
%
% Returns:
%
%   peaks: (num_peaks x 1) array containing peak ids after non-maximum 
%          suppression.
%

% INITIALIZATION
assert(mod(ws,2) == 1);
peaks = [];
% get size of heatmap
w = size(heatmap, 2); h = size(heatmap, 1);
% get number of points that will be considered
num_points = sum(heatmap(:) > peak_threshold);
[~, max_peak_ids] = sort(heatmap(:), 'descend');

mask = zeros(size(heatmap));
for idx = max_peak_ids(1:num_points)'
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

