function [frame] = box2frame(box)
% BOX2FRAME
%
% [frame] = box2frame(box)
% tranform boxes into frames (another representation).
%
% Parameters:
%
%		box: (4 x K) matrix where K is the number of proposals.
%
% Returns:
%
%		frame: (6 x K) matrix, each column contains info of a box
%
%			frame(1:2, i): center (x,y) of the proposals i.
%			frame(3, i): half the width of the proposals i.
%			frame(6, i): half the height of the proposals i.
%			frame(4:5, i): zeros.
%

[~,K] = size(box) ;
frame = zeros(6,K) ;

%[xc; yc]
frame(1:2,:) = [(box(3, :) + box(1, :)) ./ 2; ...
                (box(4, :) + box(2, :)) ./ 2];
frame(3,:) = box(3, :) - frame(1,:); % half width
frame(6,:) = box(4, :) - frame(2,:); % half height
end