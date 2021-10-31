% SUHA KWAK, post-doctoral researcher, WILLOW - INRIA/ENS



function img = drawBox(img, bbox, blwidth, blcolor)


if sum(bbox) <= 0
	return;
end	

[H, W, ~] = size(img);


% bbox = [xmin, ymin, xmax, ymax]
bbox = max(bbox, 1);
xmin = min(W, bbox(1));
xmax = min(W, bbox(3));
ymin = min(H, bbox(2));
ymax = min(H, bbox(4));
% xmin = min(W, bbox(1));
% xmax = min(W, bbox(1) + bbox(3));
% ymin = min(H, bbox(2));
% ymax = min(H, bbox(2) + bbox(4));

lstep = floor((blwidth-1) / 2);

xmin_lb = max(1, xmin-lstep);
xmin_ub = min(W, xmin-lstep+blwidth);
xmax_lb = max(1, xmax-lstep);
xmax_ub = min(W, xmax-lstep+blwidth);
ymin_lb = max(1, ymin-lstep);
ymin_ub = min(H, ymin-lstep+blwidth);
ymax_lb = max(1, ymax-lstep);
ymax_ub = min(H, ymax-lstep+blwidth);


img(ymin_lb : ymin_ub, xmin : xmax, 1) = blcolor(1);
img(ymin_lb : ymin_ub, xmin : xmax, 2) = blcolor(2);
img(ymin_lb : ymin_ub, xmin : xmax, 3) = blcolor(3);

img(ymax_lb : ymax_ub, xmin : xmax, 1) = blcolor(1);
img(ymax_lb : ymax_ub, xmin : xmax, 2) = blcolor(2);
img(ymax_lb : ymax_ub, xmin : xmax, 3) = blcolor(3);

img(ymin : ymax, xmin_lb : xmin_ub, 1) = blcolor(1);
img(ymin : ymax, xmin_lb : xmin_ub, 2) = blcolor(2);
img(ymin : ymax, xmin_lb : xmin_ub, 3) = blcolor(3);

img(ymin : ymax, xmax_lb : xmax_ub, 1) = blcolor(1);
img(ymin : ymax, xmax_lb : xmax_ub, 2) = blcolor(2);
img(ymin : ymax, xmax_lb : xmax_ub, 3) = blcolor(3);


