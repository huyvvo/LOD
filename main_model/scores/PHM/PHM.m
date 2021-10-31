function [confidenceMap] = PHM(viewA, viewB, opt)
% PHM
%
% [confidenceMap] = PHM(viewA, viewB, opt)
% Efficient Hough matching written by Minsu Cho, Inria - WILLOW
% Modified and documented by Huy V. Vo, Inria/ENS - Valeo.ai
%
% Parameters:
%
%       viewA, viewB: structs respectively coressponding to imageA 
%           and imageB created by the load_view function 
%           in han_code/utils/commonFunctions.
%
%           They contain the following fields:
%               
%               frame: (6 x N) containing info of each proposal.
%                   See 'box2frame' for details. 
%
%               bbox: the rectangle represents the whole image.
%
%               img: image, 3 dimensional (RGB) image.
%
%               desc: array of size (N x feat_dim) containing 
%                   features for each proposal, feat_dim is the
%                   length of each feature.
%
%          
%
%       opt: struct containing other parameters of the PHM 
%           algorithm. They are:
%
%           bDensityAware:
%
%           bDeleteByAspect:
%
%           bSimVote:
%
%           feature:
%
% Returns:
%
%       confidenceMap: a matrix containing confidence score of 
%           matches between imageA and imageB.
%     

% -------------------------------------------------------------------------
% SETTING PARAMETERS

bVerbose = false;
bDensityAware = opt.bDensityAware;
bDeleteByAspect = opt.bDeleteByAspect;
bSimVote = opt.bSimVote;
bFeature = opt.feature;
thresAspect = 2.0;

if bDensityAware
  po = 4;    method = 'circles';
  radius = [];    N = [];    n = [];    ms = 5;
  figure; axis ij;
  out = scatplot(viewA.frame(1,:), viewA.frame(2,:), ...
                 method,radius,N,n,po,ms);
  fdensity = out.dd';
else
  fdensity = ones(size(viewA.frame,2),1)';
end

%----------------------------------------------
% LEARNING HOUGH TRANSFORM

% compute inverted frames and determinants
% invframe is 
% detA contains the area of the proposals
[invframe, detA]  = inv_frame(viewA.frame);

% project the points on the standard frame
ptRef = viewA.bbox(3:4); % bottom right point
xTrain = invframe(3,:)*ptRef(1) + invframe(5,:)*ptRef(2) + ...
         invframe(1,:);
yTrain = invframe(4,:)*ptRef(1) + invframe(6,:)*ptRef(2) + ...
         invframe(2,:);
sTrain = detA; % area of the proposals / original scale
clear invframe detA;

aTrain = viewA.frame(3,:)./viewA.frame(6,:);% aspect ratio
aTest = viewB.frame(3,:)./viewB.frame(6,:);

%% inference Hough transform
% setting for Hough space bins
width = size(viewB.img, 2) * 2;
height = size(viewB.img, 1) * 2;
nCell = 8000;  % cells in x-y space
szCell = max(sqrt((width*height)/nCell), 5);

% identify the number of cells to compute the histogram
% each cell correspond to a cartesian product of intervals in
% x, y and scale space.
model.nSpatialX = round(width/szCell);
model.nSpatialY = round(height/szCell);
model.nSpatialS = 20;
dimBin = [model.nSpatialY, model.nSpatialX, model.nSpatialS];

% initialize histogram
histo = zeros(prod(dimBin), 1, 'single');
histo_add = zeros(prod(dimBin), 1, 'single');
histo_count = zeros(prod(dimBin), 1, 'single');
% create bins in x, y and scale space delimited by elements of
% xbinv, ybinv and sbinv respectively
xbinv = linspace(1, width, dimBin(2)+1);
ybinv = linspace(1, height, dimBin(1)+1);
sbinv = linspace(-4, 4, dimBin(3)+1); % log_2 scale

%---------------------------------------------------------------------
tic;
if strcmp(bFeature,'HOG_LDA')
  % use Whiten HOG as in 
  % Discriminative Decorrelation for Clustering and Classification!
  % Hariharan 2012
  load('bg11.mat');
  nY = 8; nX = 8;
  % learn the LDA model, ref. the paper for details.
  % bg.R: rotation matrix
  % bg.mu_bg: 'mean background', \mu_0
  [bg.R, bg.mu_bg] = whiten(bg,nX,nY);
  % compute full sim matrix efficiently
  % whiten features are computed as R^-1*(x-mu_bg) where
  % x is the matrix of raw features
  A = viewA.desc - repmat(bg.mu_bg, 1, size(viewA.desc, 2));
  A = bg.R\(bg.R'\A);
  B = viewB.desc;
  bias = -A' * bg.mu_bg;
  wVoteFull = [A; bias']' * [B; ones(1, size(B, 2))];
  wVoteFull = max(wVoteFull, 0);
  fprintf('wdot HOG_LDA\n');
%---------------------------------------
elseif strcmp(bFeature, 'RAW')
  A = viewA.desc;
  B = viewB.desc;
  wVoteFull = A'*B;
  wVoteFull = max(wVoteFull, 0);
  fprintf('wdot RAW\n');
%---------------------------------------
elseif strcmp(bFeature,'SPM')
  % chi2 kernel embedding (inner product)
  wVoteFull = vl_alldist2(sparse(double(viewA.desc)), ...
                          sparse(double(viewB.desc)), 'KCHI2');
  fprintf('chi2\n');
else
  error('Feature not supported!');
end
fprintf('Size of wVoteFull is: (%d x %d)\n', size(wVoteFull, 1), ...
                                           size(wVoteFull, 2));
toc;
%--------------------------------------------------------------------
tic;
tableBina = zeros(size(viewA.frame, 2), size(viewB.frame, 2), ...
                  'uint32');

if bVerbose
  fprintf('vote weight: max %f - min %f\n', max(wVoteFull(:)), ...
                                            min(wVoteFull(:)));
end

nFeatTrain = size(viewA.desc, 2);
nFeatTest = size(viewB.desc, 2);

% do voting
for j=1:nFeatTrain
  xVote = viewB.frame(3,:)*xTrain(j) + ...
          viewB.frame(5,:)*yTrain(j) + viewB.frame(1,:);
  yVote = viewB.frame(4,:)*xTrain(j) + ...
          viewB.frame(6,:)*yTrain(j) + viewB.frame(2,:);
  % original scale
  sVote = abs((viewB.frame(3,:).*viewB.frame(6,:) - ...
          viewB.frame(5,:).*viewB.frame(4,:))./sTrain(j)); 
  sVote = log(sVote)./log(2);
  if bSimVote
    wVoteTmp = wVoteFull(j,:)./fdensity(j);
  else
    wVoteTmp = ones(1,nFeatTest)./fdensity(j);
  end
  
  xBin = vl_binsearch(xbinv, xVote);
  yBin = vl_binsearch(ybinv, yVote);
  sBin = vl_binsearch(sbinv, sVote);
  bValid = (xBin >= 1) & (xBin <= dimBin(2)) ...
      & (yBin >= 1) & (yBin <= dimBin(1)) ...
      & (sBin >= 1) & (sBin <= dimBin(3));
  
  if bDeleteByAspect
    bValid = bValid & (aTest/aTrain(j) < thresAspect) & ...
             (aTest/aTrain(j) > 1/thresAspect);
  end
  
  idxValid = find(bValid);
  if ~isempty(idxValid)
    binsx = xBin(idxValid);%[ binsx, xBin(idxValid) ];
    binsy = yBin(idxValid);%[ binsy, yBin(idxValid) ];
    binss = sBin(idxValid);%[ binss, sBin(idxValid) ];
    wVote = wVoteTmp(idxValid);%[ wVote, wVoteTmp(idxValid) ];
    bins = binsy + (binsx-1)*dimBin(1) + ...
           (binss-1)*dimBin(1)*dimBin(2);
    
    nBins = numel(bins);
    for p=1:nBins
      histo_add(bins(p)) = max( histo_add(bins(p)), wVote(p));
    end
    histo = histo + histo_add;
    % histo_count(histo_add > 0) = histo_count(histo_add > 0) + 1;
    histo_add(bins) = 0; % make it zeros again
    
    % store bin ids
    tableBina(j, idxValid) = bins;
  end
end
toc;
% fprintf('%d ', histo_count(histo_count > 0));
% fprintf('Mean matches: %.2f, max matches: %d\n', mean(histo_count(histo_count > 0)), max(histo_count)); 
houghspace = reshape(histo, [model.nSpatialY, model.nSpatialX, ...
                             model.nSpatialS]);

% smoothen by convolution
skernel = fspecial3('gaussian', [5 5 5]);
houghspace = convn(houghspace, skernel,'same');


if bVerbose
  % show Hough spaces by scales
  figure;
  for jj=1:model.nSpatialS
    subplot(5,5,jj);  imagesc(houghspace(:,:,jj));
    title([ 'max hits: ' num2str(max(max(houghspace(:,:,jj)))) ]);
  end
  pause;
end

%% make the Hough space confidence
% make the confidence map (# of frames A by # of frames B)
confidenceMap = zeros(size(viewA.frame, 2), ...
                      size(viewB.frame, 2),'single');
idxValid = find(tableBina>0);
confidenceMap(idxValid) = wVoteFull(idxValid) .* ...
                          houghspace(tableBina(idxValid));
% confidenceMap(idxValid) = (confidenceMap(idxValid) + 1).*houghspace(tableBina(idxValid));
end

