function [ standout ] = standout_box_pair_v4(boxesA, boxesB, confidence, max_iteration, area_ratio)
% STANDOUT_BOX_PAIR_V4
% 
% [ standout ] = standout_box_pair_v4(boxesA, boxesB, confidence, max_iteration, area_ratio)
%
% Compute the exact standout score of pairs of boxes.
% Given a pair of boxes (boxa, boxb), the standout score of (a,b) is defined as
%           confidence(boxa, boxb) - max_{c \in parent(boxa), d \in parent(boxb)} confidence(c,d)
% where parent(h) of a box h is the set of boxes in the same images satisfying some conditions
% specified by variable 'area_ratio', 'intersection_ratio', 'iou' and 'quant'.
%
% Parameters:
%
%       boxesA: 4xN matrix where N is the number of boxes in A.
%
%       boxesB: 4xM matrix where M is the number of boxes in B.
%
%       confidence: NxM matrix, confidence of matches between boxes in A and B.
%
%       max_iteration:
%
%       area_ratio: a parameter to define parenthood of proposals. Set it to 0.8
%                   for Object Discovery and 0.5 for VOC.
%
% Returns:
%
%       standout: NxM matrix, the standout score of matches between boxes in A and B.
%

%------------------------------------------------------------------------
% SET DEFAULTS PARAMETERS

if ~exist('max_iteration', 'var')
    max_iteration = 5000;
end
max_iteration = min(max_iteration, numel(confidence(:)));

%------------------------------------------------------------------------
% TRANSFORM BOXES TO RECTANGLES (MAKE TOP-LEFT POINTS (0,0))
% AND COMPUTE AREA OF BOXES.

addpath(genpath('../../utils/commonFunctions/'));

rectA = box2rect(boxesA); % make the top left point (0,0)
rectB = box2rect(boxesB); % make the top left point (0,0)
areaA = rectA(3,:).*rectA(4,:); % compute area of boxes
areaB = rectB(3,:).*rectB(4,:); % compute area of boxes

rmpath(genpath('../../utils/commonFunctions/'));

%------------------------------------------------------------------------
% SET PARAMETERS FOR DEFINING THE INCLUSION RELATION BETWEEN BOXES

% area_ratio = 0.5; % for VOC
% area_ratio = 0.8; % for ObjectDiscovery
intersection_ratio = 0.8;
iou = 0.8;
quant = 0.95;

%----------------------------------------------------------------------------
% FIND PARENT BOXES OF EACH BOX (PARENTS OF A BOX = BOXES CONTAINING IT)

containmatA = false(size(boxesA,2)); 
containmatB = false(size(boxesB,2));

% find larger boxes containing each box in A
for k=1:size(boxesA,2)
    % choose boxes satisfying the condition on the area of parent boxes
    candidate_id = find(area_ratio * areaA > areaA(k));
    % compute the area of intersection between the current box and candidate boxes
    area_int = rectint(rectA(:,k)',rectA(:,candidate_id)');
    % find indices of boxes satisfying the condition on the area of intersection
    parent_id = candidate_id(find(area_int > intersection_ratio*areaA(k)));
    % mark parent boxes
    containmatA(k,parent_id) = true;
end

% find larger boxes containing each box in B
for k=1:size(boxesB,2)
    % choose boxes satisfying the condition on the area of parent boxes
    candidate_id = find(area_ratio * areaB > areaB(k));
    % compute the area of intersection between the current box and candidate boxes
    area_int = rectint(rectB(:,k)',rectB(:,candidate_id)');
    % find indices of boxes satisfying the condition on the area of intersection
    parent_id = candidate_id(find(area_int > intersection_ratio*areaB(k)));
    % mark parent boxes
    containmatB(k,parent_id) = true;
end

% % find the largest box in each image
% [ ~, id_largestA ] = max(areaA); 
% [ ~, id_largestB ] = max(areaB);
% % mark that the largest box is a parent of all other boxes
% containmatA(:,id_largestA) = true;
% containmatB(:,id_largestB) = true;

% compute 2 cells each contains ids of parents of each proposal in each image
parentA = cell(size(boxesA, 2), 1);
parentB = cell(size(boxesB, 2), 1);
for i = 1:size(parentA, 1)
    parentA{i} = find(containmatA(i, :));
end
for i = 1:size(parentB, 1)
    parentB{i} = find(containmatB(i, :));
end

%---------------------------------------------------------------------------------
% COMPUTE STANDOUT SCORE FOR PAIRS OF BOXES FROM CONFIDENCE

% create the matrix containing the max parent match of each match
max_parent_match = zeros(size(confidence),'single'); 

%---------------------------------------------------------------- 
% A heuristic to quickly compute the max match containing a match.
% The match (a,b) is said to contain the match (c,d) if a is a parent of c and b is
% a parent of d.
% Instead of compute the max match containing each match, we compute a list of 'siginificant'
% matches that are likely to contain other matches. Finally, for a given match, the max
% match containing it is either one of the siginificant match if one of them contains the 
% match or the match of two largest boxes in A and B. 

% get the pair of boxes with highest confidence
[ ~, max_idx_list ] = sort(confidence(:), 'descend');

for idx_ = max_iteration:-1:1
    [idA,idB] = ind2sub(size(confidence), max_idx_list(idx_));
    if confidence(idA, idB) == 0
        continue;
    end
    children_idA = containmatA(:,idA);
    children_idB = containmatB(:,idB);
    max_parent_match(children_idA,children_idB) = ...
                                confidence(idA, idB);
end

disp(sprintf('%.2f%% of elements processed ...', ...
    sum(confidence(:) == 0 | max_parent_match(:) > 0) ...
    /numel(confidence(:))*100));

for i = 1:size(boxesA,2)
    for j = 1:size(boxesB,2)
        if numel(parentA{i}) > 0 & numel(parentB{j}) > 0 & ...
           max_parent_match(i,j) == 0 & confidence(i,j) > 0
            max_parent_match(i,j) = max(max(confidence(parentA{i}, parentB{j})));
        end
    end
end

standout = confidence - max_parent_match;
standout = standout .* (standout > 0);

end