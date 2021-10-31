function [] = visualize_result(imdb, save_path, x)
% VISUALIZE_RESULT
%
% visualize_result(imdb, save_path, x)
%
% Parameters:
%
%   imdb: string or struct, path to the imdb or the imdb itself,
%         containing fields 'images', 'proposals', 'bboxes'
%
%   save_path: string, folder to save visualization
%
%   x: (n x 1) cell representing retained proposals 
%               before EM

%-----------------------------------------------------------------
% CREATE IMAGES

lw1 = 6;
lw2 = 6;
cl1 = [255,255,0];
cl2 = [255,0,0];

if strcmp(class(imdb), 'char') || strcmp(class(imdb), 'string')
  imdb = load(imdb);
end
n = size(imdb.images, 1);
mkdir(save_path);
for i = 1:n
  img_box = imdb.images{i};
  imwrite(img_box, fullfile(save_path, ...
                    sprintf('%d_ori.jpg', i)), 'jpg');

  bboxes = imdb.bboxes{i};
  for idx = 1:size(bboxes, 1)
    bbox = bboxes(idx, :);
    img_box = drawBox(img_box, bbox, lw1, [0, 0, 0]);
    img_box = drawBox(img_box, bbox, lw2, cl1);
  end
  imwrite(img_box, fullfile(save_path, ...
                    sprintf('%d_ori_gtboxes.jpg', i)), 'jpg');
  
  topnu_img_box = img_box;
  topnu_bboxes = imdb.proposals{i}(x{i}, :);
  for idx = 1:size(topnu_bboxes, 1)
    bbox = topnu_bboxes(idx, :);
    topnu_img_box = drawBox(topnu_img_box, bbox, lw1, [0,0,0]);
    topnu_img_box = drawBox(topnu_img_box, bbox, lw2, cl2);
  end
  imwrite(topnu_img_box, fullfile(save_path, ...
                          sprintf('%d_topnu.jpg', i)), 'jpg');

end

%----------------------------------------------------------------
% CREATE WEBPAGE

iwidth_img  = 280;
iheight_img = 200;

fout = fopen(fullfile(save_path, 'index.html'), 'w');
fprintf(fout, '<html><head><title>Visualization</title></head>\n');
fprintf(fout, '<br><br><br>\n');

for i = 1:n
  % start table
  fprintf(fout, '<table border="0">\n');
  fprintf(fout, '<tr>\n');

  % original image
  img_name = fullfile(sprintf('%d_ori.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, sprintf('<font size=5>Original image %d</font>', i));
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % original image with ground truth boxes
  img_name = fullfile(sprintf('%d_ori_gtboxes.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>GT boxes</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % LOD boxes
  img_name = fullfile(sprintf('%d_topnu.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>GT and LOD boxes</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % end table
  fprintf(fout, '</tr>\n');
  fprintf(fout, '</table>\n');
  fprintf(fout, '<br><br><br>\n');
end
fprintf(fout, '</html>\n');
fclose(fout);
