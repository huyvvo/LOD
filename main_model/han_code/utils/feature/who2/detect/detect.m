function [boxes, feats] = detect(im,model,thresh,return_feats)
% Detect model in images
if(nargin<4)
	return_feats=0;
end
feats=[];
pyra     = featpyramid(im,model);
interval = model.interval;
levels   = 1:length(pyra.feat);
boxes    = zeros(10000,5);
filters  = {model.w};

padx = pyra.padx;
pady = pyra.pady;
sizx = size(model.w,2);
sizy = size(model.w,1);
cnt  = 0;

% Ignore the hallucinated (inteprolated) scales
levels = levels(interval+1:end);
t=0;
for l = levels,
  scale = pyra.scale(l);
  resp  = fconv(pyra.feat{l},filters,1,1);
  resp  = resp{1};
  
  [y,x] = find(resp >= thresh);
  I  = (x-1)*size(resp,1)+y;
  if(~isempty(I) & return_feats)
  tic
  f1=zeros(sizx*sizy*(size(model.w,3)),numel(y));
  for k=1:numel(y)
	f2=pyra.feat{l}(y(1):y(1)+sizy-1,x(1):x(1)+sizx-1,:);
	%remove last feature
	%f2=f2(:,:,1:end-1);
	f1(:,k)=f2(:);
  end
  feats=[feats f1];
  t=t+toc;
  end

  x1 = (x-1-padx)*scale+1;
  y1 = (y-1-pady)*scale+1;
  x2 = x1 + sizx*scale - 1;
  y2 = y1 + sizy*scale - 1;

  i = cnt+1:cnt+length(I);
  boxes(i,:) = [x1 y1 x2 y2 resp(I)];
  cnt = cnt+length(I);
end

boxes = boxes(1:cnt,:);

