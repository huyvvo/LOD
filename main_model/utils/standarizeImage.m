function im = standarizeImage(im, height)
% resize image into the standard size
if nargin < 2
    height = 480;
end
%im = im2single(im);
if size(im,1) > height, im = imresize(im, [height NaN], 'bilinear') ; end